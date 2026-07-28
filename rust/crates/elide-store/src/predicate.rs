//! Predicates that prune before they filter.
//!
//! The point is not "WHERE works" — Arrow can filter anything once it is
//! decoded. The point is deciding NOT to read: a predicate is pushed
//! through three layers, each cheaper than the next one it saves,
//!
//!   1. the transaction log's per-file zone maps   (no I/O at all)
//!   2. Parquet row-group statistics from the footer
//!   3. Parquet's PAGE index — min/max per page inside a row group,
//!      so a surviving row group still reads only the pages that can
//!      contain a match
//!
//! and only then does the surviving data get an exact Arrow filter.
//! Layer 3 is what makes a columnar file behave like an index for
//! selective queries, and it is the layer the engine was missing.

use anyhow::{bail, Context, Result};
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::page_index::offset_index::OffsetIndexMetaData;
use parquet::file::statistics::Statistics;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

impl Value {
    fn parse(raw: &str) -> Value {
        let t = raw.trim();
        if (t.starts_with('\'') && t.ends_with('\'') && t.len() >= 2)
            || (t.starts_with('"') && t.ends_with('"') && t.len() >= 2)
        {
            return Value::Str(t[1..t.len() - 1].to_string());
        }
        match t.to_ascii_lowercase().as_str() {
            "true" => return Value::Bool(true),
            "false" => return Value::Bool(false),
            _ => {}
        }
        if let Ok(i) = t.parse::<i64>() {
            return Value::Int(i);
        }
        if let Ok(f) = t.parse::<f64>() {
            return Value::Float(f);
        }
        Value::Str(t.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Op {
    fn parse(s: &str) -> Option<Op> {
        Some(match s {
            "=" | "==" => Op::Eq,
            "!=" | "<>" => Op::Ne,
            "<" => Op::Lt,
            "<=" => Op::Le,
            ">" => Op::Gt,
            ">=" => Op::Ge,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Predicate {
    pub column: String,
    pub op: Op,
    pub value: Value,
}

impl Predicate {
    /// `"packet_size > 10000"`, `"stream = 'cam/file-129'"`,
    /// `"keyframe = true"`. Conjunctions are supplied as a list; OR is
    /// deliberately absent — it cannot prune, and a predicate that
    /// cannot prune belongs in the Arrow filter, not here.
    pub fn parse(s: &str) -> Result<Predicate> {
        // longest operators first so ">=" never lexes as ">"
        for opstr in ["<=", ">=", "!=", "<>", "==", "=", "<", ">"] {
            if let Some(i) = s.find(opstr) {
                let col = s[..i].trim();
                let val = s[i + opstr.len()..].trim();
                if col.is_empty() || val.is_empty() {
                    continue;
                }
                return Ok(Predicate {
                    column: col.to_string(),
                    op: Op::parse(opstr).context("bad operator")?,
                    value: Value::parse(val),
                });
            }
        }
        bail!("cannot parse predicate {s:?} (expected `column OP value`)")
    }

    /// Could a range [min, max] contain a row satisfying this? A `false`
    /// here is what elides bytes; `true` only means "cannot rule out".
    fn range_may_match(&self, min: &Value, max: &Value) -> bool {
        let (lo, hi, v) = match (min, max, &self.value) {
            (Value::Int(a), Value::Int(b), Value::Int(c)) => {
                (*a as f64, *b as f64, *c as f64)
            }
            (Value::Float(a), Value::Float(b), Value::Float(c)) => (*a, *b, *c),
            (Value::Int(a), Value::Int(b), Value::Float(c)) => {
                (*a as f64, *b as f64, *c)
            }
            (Value::Float(a), Value::Float(b), Value::Int(c)) => {
                (*a, *b, *c as f64)
            }
            (Value::Bool(a), Value::Bool(b), Value::Bool(c)) => (
                *a as u8 as f64,
                *b as u8 as f64,
                *c as u8 as f64,
            ),
            (Value::Str(a), Value::Str(b), Value::Str(c)) => {
                // lexicographic bounds work for the ordered operators
                return match self.op {
                    Op::Eq => a.as_str() <= c.as_str() && c.as_str() <= b.as_str(),
                    Op::Ne => !(a == b && a == c),
                    Op::Lt => a.as_str() < c.as_str(),
                    Op::Le => a.as_str() <= c.as_str(),
                    Op::Gt => b.as_str() > c.as_str(),
                    Op::Ge => b.as_str() >= c.as_str(),
                };
            }
            _ => return true, // types disagree: cannot prune, must read
        };
        match self.op {
            Op::Eq => lo <= v && v <= hi,
            Op::Ne => !(lo == hi && lo == v),
            Op::Lt => lo < v,
            Op::Le => lo <= v,
            Op::Gt => hi > v,
            Op::Ge => hi >= v,
        }
    }
}

fn stats_bounds(s: &Statistics) -> Option<(Value, Value)> {
    Some(match s {
        Statistics::Int32(v) => (
            Value::Int(*v.min_opt()? as i64),
            Value::Int(*v.max_opt()? as i64),
        ),
        Statistics::Int64(v) => {
            (Value::Int(*v.min_opt()?), Value::Int(*v.max_opt()?))
        }
        Statistics::Float(v) => (
            Value::Float(*v.min_opt()? as f64),
            Value::Float(*v.max_opt()? as f64),
        ),
        Statistics::Double(v) => {
            (Value::Float(*v.min_opt()?), Value::Float(*v.max_opt()?))
        }
        Statistics::Boolean(v) => {
            (Value::Bool(*v.min_opt()?), Value::Bool(*v.max_opt()?))
        }
        Statistics::ByteArray(v) => (
            Value::Str(v.min_opt()?.as_utf8().ok()?.to_string()),
            Value::Str(v.max_opt()?.as_utf8().ok()?.to_string()),
        ),
        _ => return None,
    })
}

/// Leaf column index for a top-level column name.
pub fn leaf_of(
    descr: &parquet::schema::types::SchemaDescriptor,
    name: &str,
) -> Option<usize> {
    (0..descr.num_columns()).find(|&i| descr.column(i).path().parts()[0] == name)
}

/// Layer 2: can this row group contain a match?
pub fn row_group_may_match(
    rg: &RowGroupMetaData,
    descr: &parquet::schema::types::SchemaDescriptor,
    preds: &[Predicate],
) -> bool {
    for p in preds {
        let Some(leaf) = leaf_of(descr, &p.column) else {
            continue; // unknown column: the Arrow filter will reject it
        };
        let Some(stats) = rg.column(leaf).statistics() else {
            continue; // no statistics: cannot prune
        };
        let Some((min, max)) = stats_bounds(stats) else {
            continue;
        };
        if !p.range_may_match(&min, &max) {
            return false;
        }
    }
    true
}

/// Layer 3: which PAGES of this row group can contain a match?
/// Returns row ranges (relative to the row group) worth reading, or
/// None when the page index cannot answer and the whole group stands.
pub fn page_ranges(
    col_index: &ColumnIndexMetaData,
    offset_index: &OffsetIndexMetaData,
    rg_rows: usize,
    pred: &Predicate,
) -> Option<Vec<(usize, usize)>> {
    let locations = offset_index.page_locations();
    let n = locations.len();
    // arrow-rs exposes per-page bounds as min_value(i)/max_value(i)
    let bounds: Vec<Option<(Value, Value)>> = match col_index {
        ColumnIndexMetaData::INT32(ix) => (0..n)
            .map(|i| {
                Some((
                    Value::Int(*ix.min_value(i)? as i64),
                    Value::Int(*ix.max_value(i)? as i64),
                ))
            })
            .collect(),
        ColumnIndexMetaData::INT64(ix) => (0..n)
            .map(|i| {
                Some((Value::Int(*ix.min_value(i)?), Value::Int(*ix.max_value(i)?)))
            })
            .collect(),
        ColumnIndexMetaData::DOUBLE(ix) => (0..n)
            .map(|i| {
                Some((
                    Value::Float(*ix.min_value(i)?),
                    Value::Float(*ix.max_value(i)?),
                ))
            })
            .collect(),
        ColumnIndexMetaData::FLOAT(ix) => (0..n)
            .map(|i| {
                Some((
                    Value::Float(*ix.min_value(i)? as f64),
                    Value::Float(*ix.max_value(i)? as f64),
                ))
            })
            .collect(),
        ColumnIndexMetaData::BOOLEAN(ix) => (0..n)
            .map(|i| {
                Some((
                    Value::Bool(*ix.min_value(i)?),
                    Value::Bool(*ix.max_value(i)?),
                ))
            })
            .collect(),
        ColumnIndexMetaData::BYTE_ARRAY(ix) => (0..n)
            .map(|i| {
                Some((
                    Value::Str(String::from_utf8(ix.min_value(i)?.to_vec()).ok()?),
                    Value::Str(String::from_utf8(ix.max_value(i)?.to_vec()).ok()?),
                ))
            })
            .collect(),
        _ => return None,
    };

    let mut out = Vec::new();
    for (i, b) in bounds.iter().enumerate() {
        let first = locations[i].first_row_index as usize;
        let last = if i + 1 < n {
            locations[i + 1].first_row_index as usize
        } else {
            rg_rows
        };
        match b {
            // a page with no bounds (all null, or an unsupported type)
            // must be read: absence of evidence is not evidence
            None => out.push((first, last)),
            Some((min, max)) => {
                if pred.range_may_match(min, max) {
                    out.push((first, last));
                }
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_forms() {
        let p = Predicate::parse("packet_size > 10000").unwrap();
        assert_eq!(p.column, "packet_size");
        assert_eq!(p.op, Op::Gt);
        assert_eq!(p.value, Value::Int(10000));

        let p = Predicate::parse("stream = 'cam/file-129'").unwrap();
        assert_eq!(p.value, Value::Str("cam/file-129".into()));

        let p = Predicate::parse("keyframe = true").unwrap();
        assert_eq!(p.value, Value::Bool(true));

        // ">=" must not lex as ">"
        let p = Predicate::parse("ts >= 5").unwrap();
        assert_eq!(p.op, Op::Ge);
        assert_eq!(p.value, Value::Int(5));
    }

    #[test]
    fn range_pruning_is_sound() {
        let p = Predicate::parse("x > 100").unwrap();
        assert!(!p.range_may_match(&Value::Int(0), &Value::Int(100)));
        assert!(p.range_may_match(&Value::Int(0), &Value::Int(101)));

        let p = Predicate::parse("x = 50").unwrap();
        assert!(p.range_may_match(&Value::Int(0), &Value::Int(100)));
        assert!(!p.range_may_match(&Value::Int(51), &Value::Int(100)));

        // a constant page equal to the excluded value is the only case
        // != can rule out
        let p = Predicate::parse("x != 7").unwrap();
        assert!(!p.range_may_match(&Value::Int(7), &Value::Int(7)));
        assert!(p.range_may_match(&Value::Int(7), &Value::Int(8)));

        let p = Predicate::parse("s = 'm'").unwrap();
        assert!(p.range_may_match(&Value::Str("a".into()), &Value::Str("z".into())));
        assert!(!p.range_may_match(&Value::Str("n".into()), &Value::Str("z".into())));
    }
}
