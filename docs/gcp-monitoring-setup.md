# GCP Monitoring and Observability Implementation Guide

This guide provides instructions for setting up comprehensive monitoring and observability for the NeuraShield application on Google Cloud Platform (GCP).

## Overview

The following GCP services will be implemented:

1. **Cloud Logging** - For centralized log management
2. **Cloud Monitoring** - For metrics collection and dashboards
3. **Cloud Trace** - For distributed tracing
4. **Cloud Profiler** - For performance analysis
5. **Cloud Error Reporting** - For error tracking

## Prerequisites

- GCP project with billing enabled
- GKE cluster set up and running
- `gcloud` CLI installed and configured
- `kubectl` CLI installed and configured to connect to your GKE cluster
- Administrative access to the GCP project

## Implementation Steps

### 1. Enable Required GCP APIs

```bash
# Run this command from your development environment
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable cloudtrace.googleapis.com
gcloud services enable cloudprofiler.googleapis.com
gcloud services enable clouderrorreporting.googleapis.com
```

### 2. Create Service Account

Create a dedicated service account for monitoring:

```bash
# Set your project ID
export PROJECT_ID=$(gcloud config get-value project)

# Create service account
gcloud iam service-accounts create neurashield-monitoring-sa \
    --display-name="NeuraShield Monitoring Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:neurashield-monitoring-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:neurashield-monitoring-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:neurashield-monitoring-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/stackdriver.resourceMetadata.writer"

# Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=neurashield-monitoring-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

### 3. Create Kubernetes Secret for the Service Account Key

```bash
# Create the monitoring namespace if it doesn't exist
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Create a secret with the service account key
kubectl create secret generic google-cloud-key --from-file=key.json -n monitoring

# Create a secret in neurashield namespace as well
kubectl create secret generic google-cloud-key --from-file=key.json -n neurashield
```

### 4. Update the GCP ConfigMap

The `k8s/gcp-configmap.yaml` file has been configured with your GCP project ID:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gcp-config
  namespace: neurashield
data:
  project_id: "supple-defender-458307-i7"  # Your GCP project ID
  service_version: "1.0.0"
  gcp_region: "us-central1"  # Update this with your preferred GCP region
  cloud_trace_sampling_rate: "1.0"  # 100% sampling rate for development, reduce for production
  profiler_enabled: "true"
  error_reporting_enabled: "true"
```

If you need to modify the region or other settings, edit this file and apply it:

```bash
kubectl apply -f k8s/gcp-configmap.yaml
```

### 5. Deploy the GCP Monitoring Components

```bash
# Apply the GCP monitoring configuration
kubectl apply -f k8s/gcp-monitoring.yaml
```

### 6. Install Dependencies in the Backend Application

```bash
cd backend
npm install @google-cloud/error-reporting @google-cloud/opentelemetry-cloud-trace-exporter @google-cloud/profiler @opentelemetry/instrumentation @opentelemetry/instrumentation-express @opentelemetry/instrumentation-http @opentelemetry/resources @opentelemetry/sdk-trace-base @opentelemetry/sdk-trace-node @opentelemetry/semantic-conventions
```

### 7. Update Backend Deployment

Update the backend deployment to include the necessary environment variables:

```bash
kubectl apply -f k8s/backend-deployment.yaml
```

### 8. Run the Setup Script

Execute the setup script to provision GCP monitoring dashboards and alerts:

```bash
bash /home/jub/Cursor/neurashield/scripts/setup-gcp-monitoring.sh
```

### 9. Verify Monitoring Setup

#### Cloud Logging

1. Go to GCP Console > Logging > Logs Explorer
2. Filter logs by resource: `k8s_container` and namespace: `neurashield`
3. Verify logs are being ingested

#### Cloud Monitoring

1. Go to GCP Console > Monitoring > Dashboards
2. Look for the "NeuraShield Overview" dashboard
3. Verify metrics are being displayed

#### Cloud Trace

1. Go to GCP Console > Trace > Trace List
2. Verify traces from your application are being captured

#### Cloud Profiler

1. Go to GCP Console > Profiler
2. Select your service from the dropdown
3. View CPU and heap profiles

#### Cloud Error Reporting

1. Go to GCP Console > Error Reporting
2. Verify any errors from your application are being reported

## Custom Metrics Setup

The application has the following custom metrics already configured:

- `http_requests_total` - Total number of HTTP requests
- `http_request_duration_seconds` - Duration of HTTP requests
- `blockchain_sync_status` - Blockchain sync status (1 = synced, 0 = not synced)
- `model_accuracy` - Current accuracy of the threat detection model

These metrics are exposed on the `/metrics` endpoint and are automatically scraped by Prometheus, then exported to Cloud Monitoring.

## Setting Up Alerts

The setup script creates a basic "High Error Rate" alert. To create additional alerts:

1. Go to GCP Console > Monitoring > Alerting
2. Click "Create Policy"
3. Configure conditions, notifications, and documentation

## Next Steps

After implementing this monitoring and observability solution, consider:

1. Setting up notification channels for your alerts (email, SMS, Slack, PagerDuty, etc.)
2. Creating additional dashboards for specific use cases
3. Implementing more granular alerting policies
4. Adjusting the Cloud Trace sampling rate for production
5. Setting up SLOs (Service Level Objectives) for critical services

## Troubleshooting

### Common Issues

1. **Logs not appearing in Cloud Logging**
   - Check if the service account has the `logging.logWriter` role
   - Verify the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set correctly

2. **Metrics not appearing in Cloud Monitoring**
   - Check if the service account has the `monitoring.metricWriter` role
   - Verify that Prometheus is scraping the metrics endpoints

3. **Traces not appearing in Cloud Trace**
   - Check if the `cloudtrace.googleapis.com` API is enabled
   - Verify that the service account has the appropriate permissions

4. **Profiler not working**
   - Check if the `cloudprofiler.googleapis.com` API is enabled
   - Verify that the service is running with the correct service name and version

## References

- [GCP Logging Documentation](https://cloud.google.com/logging/docs)
- [GCP Monitoring Documentation](https://cloud.google.com/monitoring/docs)
- [GCP Trace Documentation](https://cloud.google.com/trace/docs)
- [GCP Profiler Documentation](https://cloud.google.com/profiler/docs)
- [GCP Error Reporting Documentation](https://cloud.google.com/error-reporting/docs)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) 