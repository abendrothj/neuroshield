#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Use the specific project ID
PROJECT_ID="supple-defender-458307-i7"
REGION=$(gcloud config get-value compute/region || echo "us-central1")
SERVICE_ACCOUNT="neurashield-monitoring-sa"

echo -e "${GREEN}Setting up GCP monitoring and observability for project: ${PROJECT_ID}${NC}"

# Create service account for monitoring
echo -e "${YELLOW}Creating service account for monitoring...${NC}"
gcloud iam service-accounts create ${SERVICE_ACCOUNT} \
    --display-name="NeuraShield Monitoring Service Account" \
    --project=${PROJECT_ID}

# Grant necessary permissions
echo -e "${YELLOW}Assigning roles to service account...${NC}"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/stackdriver.resourceMetadata.writer"

# Create and download a key for the service account
echo -e "${YELLOW}Creating service account key...${NC}"
gcloud iam service-accounts keys create key.json \
    --iam-account=${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com \
    --project=${PROJECT_ID}

# Enable necessary APIs
echo -e "${YELLOW}Enabling required GCP APIs...${NC}"
gcloud services enable monitoring.googleapis.com --project=${PROJECT_ID}
gcloud services enable logging.googleapis.com --project=${PROJECT_ID}
gcloud services enable cloudtrace.googleapis.com --project=${PROJECT_ID}
gcloud services enable cloudprofiler.googleapis.com --project=${PROJECT_ID}
gcloud services enable clouderrorreporting.googleapis.com --project=${PROJECT_ID}

# Set up custom dashboards
echo -e "${YELLOW}Setting up custom monitoring dashboards...${NC}"
MONITORING_DASHBOARD=$(cat << EOF
{
  "displayName": "NeuraShield Overview",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "CPU Usage",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"kubernetes.io/container/cpu/core_usage_time\" AND resource.type=\"k8s_container\" AND resource.labels.container_name=\"backend\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "yPos": 0,
        "xPos": 6,
        "widget": {
          "title": "Memory Usage",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"kubernetes.io/container/memory/used_bytes\" AND resource.type=\"k8s_container\" AND resource.labels.container_name=\"backend\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "yPos": 4,
        "widget": {
          "title": "API Request Count",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/http_requests_total\" AND resource.type=\"k8s_container\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "yPos": 4,
        "xPos": 6,
        "widget": {
          "title": "Error Rate",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/http_requests_total\" AND resource.type=\"k8s_container\" AND metric.labels.status=~\"5.*\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "width": 12,
        "height": 4,
        "yPos": 8,
        "widget": {
          "title": "Model Accuracy",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/model_accuracy\" AND resource.type=\"k8s_container\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                }
              }
            ]
          }
        }
      }
    ]
  }
}
EOF
)

# Create dashboard using gcloud API
echo "${MONITORING_DASHBOARD}" > dashboard.json
gcloud monitoring dashboards create --config-from-file=dashboard.json --project=${PROJECT_ID}

# Set up alert policies
echo -e "${YELLOW}Setting up alert policies...${NC}"
HIGH_ERROR_RATE_ALERT=$(cat << EOF
{
  "displayName": "High Error Rate Alert",
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "Error Rate > 10%",
      "conditionThreshold": {
        "filter": "metric.type=\"custom.googleapis.com/http_requests_total\" AND resource.type=\"k8s_container\" AND metric.labels.status=~\"5.*\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_RATE"
          }
        ],
        "comparison": "COMPARISON_GT",
        "thresholdValue": 0.1,
        "duration": "300s",
        "trigger": {
          "count": 1
        }
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "604800s"
  },
  "notificationChannels": []
}
EOF
)

echo "${HIGH_ERROR_RATE_ALERT}" > alert_policy.json
gcloud alpha monitoring policies create --policy-from-file=alert_policy.json --project=${PROJECT_ID}

# Save the service account key for later use with Kubernetes
echo -e "${YELLOW}Service account key saved as key.json${NC}"
echo -e "${YELLOW}To manually create Kubernetes secrets:${NC}"
echo -e "kubectl create secret generic google-cloud-key --from-file=key.json -n monitoring"
echo -e "kubectl create secret generic google-cloud-key --from-file=key.json -n neurashield"

echo -e "${GREEN}GCP monitoring and observability setup complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Set up notification channels in Cloud Monitoring console"
echo -e "2. Associate notification channels with alert policies"
echo -e "3. Configure Cloud Trace sampling rate as needed"
echo -e "4. Set up Cloud Profiler for continuous profiling" 