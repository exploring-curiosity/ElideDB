#!/usr/bin/env bash
# One-command deploy of the agent to AWS, using only always-free services.
#
#   export PRECEDENT_DSN='postgresql://...cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full'
#   ./showreel/deploy/deploy.sh
#
# What it creates:
#   Lambda function    the agent worker, scheduled               (1M req free)
#   EventBridge rule   fires the worker every 5 minutes          (free)
#
# The bucket is not created here. setup_bucket.sh owns it, because the corpus and
# the worker have nothing to do with each other: this function never opens a
# clip. It claims an episode, reads its neighbours' filings, decides, and writes.
#
# What it deliberately does NOT create: EC2, NAT gateways, API Gateway, RDS.
# Those are what turn a $0 architecture into a metered one, and on an account
# created after 2025-07-15 they draw straight down the $200 of credits.
set -euo pipefail

: "${PRECEDENT_DSN:?set PRECEDENT_DSN to the CockroachDB connection string}"
# The worker goes where its memory is, which is NOT where the bucket is. Every
# invocation is a handful of round trips to CockroachDB and zero bytes of S3, so
# it belongs in the cluster's region; the corpus belongs near whoever is watching
# the video. Read the region out of the DSN rather than trusting a default.
REGION="${LAMBDA_REGION:-$(printf '%s' "$PRECEDENT_DSN" | sed -n 's/.*\.aws-\([a-z0-9-]*\)\.cockroachlabs\.cloud.*/\1/p')}"
REGION="${REGION:-us-east-2}"
FN="${FN:-precedent-agent}"
ROLE_NAME="${ROLE_NAME:-precedent-lambda-role}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$(mktemp -d)"

echo "==> lambda in $REGION (beside the cluster), function $FN"

# ---- 1. the role ---------------------------------------------------------
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
      --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "    created role $ROLE_NAME (logs only: the agent needs nothing else)"
  sleep 12                      # IAM is eventually consistent
fi
ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"

# ---- 2. the package ------------------------------------------------------
echo "==> building"
cp "$HERE/lambda_worker.py" "$HERE/../agent.py" "$HERE/../db.py" \
   "$HERE/../schema.sql" "$BUILD/"
# manylinux wheel, because the build host is a Mac and the runtime is Linux.
pip install --quiet --platform manylinux2014_x86_64 --implementation cp \
    --python-version 3.11 --only-binary=:all: --target "$BUILD" \
    psycopg2-binary >/dev/null
# The CA cert travels with the function: verify-full is not optional against a
# cloud cluster, and Lambda has no $HOME/.postgresql to read it from.
cp "$HOME/.postgresql/root.crt" "$BUILD/root.crt"
(cd "$BUILD" && zip -qr package.zip .)
echo "    $(du -h "$BUILD/package.zip" | cut -f1) package"

# ---- 3. the function -----------------------------------------------------
DSN_WITH_CERT="${PRECEDENT_DSN%%&sslrootcert=*}&sslrootcert=/var/task/root.crt"
if aws lambda get-function --function-name "$FN" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FN" --region "$REGION" \
      --zip-file "fileb://$BUILD/package.zip" >/dev/null
  aws lambda wait function-updated --function-name "$FN" --region "$REGION"
  aws lambda update-function-configuration --function-name "$FN" --region "$REGION" \
      --environment "Variables={PRECEDENT_DSN=$DSN_WITH_CERT,PRECEDENT_ENGINE=cockroach}" >/dev/null
else
  aws lambda create-function --function-name "$FN" --region "$REGION" \
      --runtime python3.11 --role "$ROLE_ARN" --handler lambda_worker.handler \
      --timeout 120 --memory-size 512 \
      --zip-file "fileb://$BUILD/package.zip" \
      --environment "Variables={PRECEDENT_DSN=$DSN_WITH_CERT,PRECEDENT_ENGINE=cockroach}" >/dev/null
fi
echo "    deployed $FN"

# ---- 4. the schedule -----------------------------------------------------
aws events put-rule --name "${FN}-tick" --region "$REGION" \
    --schedule-expression "rate(5 minutes)" >/dev/null
FN_ARN="$(aws lambda get-function --function-name "$FN" --region "$REGION" \
          --query Configuration.FunctionArn --output text)"
aws lambda add-permission --function-name "$FN" --region "$REGION" \
    --statement-id "${FN}-tick" --action lambda:InvokeFunction \
    --principal events.amazonaws.com >/dev/null 2>&1 || true
aws events put-targets --rule "${FN}-tick" --region "$REGION" \
    --targets "Id=1,Arn=$FN_ARN" >/dev/null
echo "    scheduled every 5 minutes"

rm -rf "$BUILD"
echo
echo "==> done. drain the queue now with:"
echo "    aws lambda invoke --function-name $FN --region $REGION /dev/stdout"
