#!/bin/bash
# script pour créer les stacks et orchestrateurs
zenml artifact-store register s3_store --flavor=s3 --path=s3://${MINIO_DEFAULT_BUCKETS} --key="${MINIO_ROOT_USER}" --secret="${MINIO_ROOT_PASSWORD}" --client_kwargs="{\"endpoint_url\": \"${MINIO_SERVER_URL}\"}"
zenml stack register local_gitflow_stack -a s3_store -o default 
#zenml stack set local_gitflow_stack