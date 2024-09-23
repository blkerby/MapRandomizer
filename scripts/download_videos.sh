#!/bin/bash

export ENDPOINT=https://s3.us-west-004.backblazeb2.com

# key: public-map-rando-videos
# Backblaze does not allow unauthenticated use of the S3-compatible API even for public buckets,
# so we have to use an application key. This key allows only read-only access. It is deliberately
# checked publicly here into source control as it is intended for public access.
#
# This will download the entire bucket, including the original AVI videos (compressed with xz),
# encoded thumbnails (PNG), animated highlights (WebP), and encoded full videos (MP4), requiring
# 10 GB or more of free space on the local filesystem:
export AWS_ACCESS_KEY_ID=00489874733cab00000000004
export AWS_SECRET_ACCESS_KEY=K004bVFHb3NDPktA6ESq5myiM4awAXw
aws s3 sync s3://map-rando-videos ./map-rando-videos --endpoint ${ENDPOINT}

# key: public-map-rando-videos-pgdump
#
# This downloads the daily database backups, including video metadata.
# These are small compared to the other bucket.
export AWS_ACCESS_KEY_ID=00489874733cab00000000005
export AWS_SECRET_ACCESS_KEY=K004EwM6mky6r24fqiRATbuBuhpM540
aws s3 sync s3://map-rando-videos-pgdump ./map-rando-videos-pgdump --endpoint ${ENDPOINT}