#!/usr/bin/env bash
# Create the one private bucket the deployment needs.
#
#   BUCKET=precedent-<something-unique> ./showreel/deploy/setup_bucket.sh
#
# Private, encrypted, versioning off. The clips are reachable only through a
# presigned URL this app signs, which is the difference between "a demo that
# serves video" and "a bucket someone else can crawl". A public bucket is the
# most common way a hackathon submission leaks the data it was built on.
set -euo pipefail
: "${BUCKET:?set BUCKET to a globally-unique bucket name}"
REGION="${AWS_REGION:-$(aws configure get region)}"
REGION="${REGION:-us-east-1}"

echo "==> s3://$BUCKET in $REGION"

if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "    exists already"
else
  # us-east-1 is the one region that rejects LocationConstraint, because it is
  # the API's default and naming it is an error rather than a no-op.
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
  echo "    created"
fi

aws s3api put-public-access-block --bucket "$BUCKET" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
aws s3api put-bucket-encryption --bucket "$BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
echo "    public access blocked, SSE-S3 on"

# GET-only CORS. A <video> tag following a redirect does not need this, but a
# range request issued by JS does, and the seek bar is the first thing anyone
# touches.
aws s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration '{
  "CORSRules":[{"AllowedMethods":["GET","HEAD"],"AllowedOrigins":["*"],
                "AllowedHeaders":["Range"],
                "ExposeHeaders":["Content-Length","Content-Range","Accept-Ranges"],
                "MaxAgeSeconds":3000}]}'
echo "    CORS: GET and HEAD with Range"

cat <<EOF

    export PRECEDENT_BUCKET=$BUCKET
    export AWS_REGION=$REGION

next: .venv-libero/bin/python showreel/deploy/upload_assets.py
EOF
