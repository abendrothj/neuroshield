#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Setting up monitoring infrastructure...${NC}"

# Verify GCP service account key exists
if [ ! -f "../key.json" ]; then
    echo "Error: GCP service account key not found at ../key.json"
    echo "Please set up a GCP service account with monitoring permissions"
    exit 1
fi

# Create monitoring namespace if it doesn't exist
echo "Creating monitoring namespace..."
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Create Kubernetes secret for GCP service account
echo "Creating Kubernetes secret for GCP credentials..."
kubectl create secret generic google-cloud-key \
    --from-file=key.json=../key.json \
    --namespace=monitoring \
    --dry-run=client -o yaml | kubectl apply -f -

# Install Prometheus Operator using Helm
echo "Installing Prometheus Operator using Helm..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create custom values file for Prometheus Operator
cat <<EOF > prometheus-values.yaml
grafana:
  enabled: true
  adminPassword: NeuraShield2023!
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
      - name: 'neurashield'
        orgId: 1
        folder: 'NeuraShield'
        type: file
        disableDeletion: false
        editable: true
        options:
          path: /var/lib/grafana/dashboards/neurashield
  dashboards:
    neurashield:
      neurashield-dashboard:
        json: |
          {
            "annotations": {
              "list": []
            },
            "editable": true,
            "gnetId": null,
            "graphTooltip": 0,
            "hideControls": false,
            "links": [],
            "refresh": "10s",
            "rows": [
              {
                "collapse": false,
                "height": "250px",
                "panels": [
                  {
                    "aliasColors": {},
                    "bars": false,
                    "dashLength": 10,
                    "dashes": false,
                    "datasource": "Prometheus",
                    "fill": 1,
                    "id": 1,
                    "legend": {
                      "avg": false,
                      "current": false,
                      "max": false,
                      "min": false,
                      "show": true,
                      "total": false,
                      "values": false
                    },
                    "lines": true,
                    "linewidth": 1,
                    "links": [],
                    "nullPointMode": "null",
                    "percentage": false,
                    "pointradius": 5,
                    "points": false,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "span": 12,
                    "stack": false,
                    "steppedLine": false,
                    "targets": [
                      {
                        "expr": "sum(rate(api_requests_total[5m])) by (method, path)",
                        "format": "time_series",
                        "intervalFactor": 2,
                        "legendFormat": "{{method}} {{path}}",
                        "refId": "A"
                      }
                    ],
                    "thresholds": [],
                    "timeFrom": null,
                    "timeShift": null,
                    "title": "API Requests",
                    "tooltip": {
                      "shared": true,
                      "sort": 0,
                      "value_type": "individual"
                    },
                    "type": "graph",
                    "xaxis": {
                      "buckets": null,
                      "mode": "time",
                      "name": null,
                      "show": true,
                      "values": []
                    },
                    "yaxes": [
                      {
                        "format": "short",
                        "label": null,
                        "logBase": 1,
                        "max": null,
                        "min": null,
                        "show": true
                      },
                      {
                        "format": "short",
                        "label": null,
                        "logBase": 1,
                        "max": null,
                        "min": null,
                        "show": true
                      }
                    ]
                  }
                ],
                "repeat": null,
                "repeatIteration": null,
                "repeatRowId": null,
                "showTitle": false,
                "title": "Dashboard Row",
                "titleSize": "h6"
              },
              {
                "collapse": false,
                "height": 250,
                "panels": [
                  {
                    "aliasColors": {},
                    "bars": false,
                    "dashLength": 10,
                    "dashes": false,
                    "datasource": "Prometheus",
                    "fill": 1,
                    "id": 2,
                    "legend": {
                      "avg": false,
                      "current": false,
                      "max": false,
                      "min": false,
                      "show": true,
                      "total": false,
                      "values": false
                    },
                    "lines": true,
                    "linewidth": 1,
                    "links": [],
                    "nullPointMode": "null",
                    "percentage": false,
                    "pointradius": 5,
                    "points": false,
                    "renderer": "flot",
                    "seriesOverrides": [],
                    "spaceLength": 10,
                    "span": 12,
                    "stack": false,
                    "steppedLine": false,
                    "targets": [
                      {
                        "expr": "histogram_quantile(0.95, sum(rate(api_latency_bucket[5m])) by (le, path))",
                        "format": "time_series",
                        "intervalFactor": 2,
                        "legendFormat": "{{path}}",
                        "refId": "A"
                      }
                    ],
                    "thresholds": [],
                    "timeFrom": null,
                    "timeShift": null,
                    "title": "API Latency (p95)",
                    "tooltip": {
                      "shared": true,
                      "sort": 0,
                      "value_type": "individual"
                    },
                    "type": "graph",
                    "xaxis": {
                      "buckets": null,
                      "mode": "time",
                      "name": null,
                      "show": true,
                      "values": []
                    },
                    "yaxes": [
                      {
                        "format": "ms",
                        "label": null,
                        "logBase": 1,
                        "max": null,
                        "min": null,
                        "show": true
                      },
                      {
                        "format": "short",
                        "label": null,
                        "logBase": 1,
                        "max": null,
                        "min": null,
                        "show": true
                      }
                    ]
                  }
                ],
                "repeat": null,
                "repeatIteration": null,
                "repeatRowId": null,
                "showTitle": false,
                "title": "Dashboard Row",
                "titleSize": "h6"
              }
            ],
            "schemaVersion": 14,
            "style": "dark",
            "tags": [],
            "time": {
              "from": "now-6h",
              "to": "now"
            },
            "timepicker": {
              "refresh_intervals": [
                "5s",
                "10s",
                "30s",
                "1m",
                "5m",
                "15m",
                "30m",
                "1h",
                "2h",
                "1d"
              ],
              "time_options": [
                "5m",
                "15m",
                "1h",
                "6h",
                "12h",
                "24h",
                "2d",
                "7d",
                "30d"
              ]
            },
            "timezone": "",
            "title": "NeuraShield API Dashboard",
            "version": 1
          }
prometheus:
  enabled: true
  serviceMonitorNamespaceSelector: {}
  serviceMonitorSelector: {}
alertmanager:
  enabled: true
  baseURL: /alertmanager
  alertmanagerSpec:
    externalUrl: /alertmanager
EOF

# Check if prometheus-operator is already installed
if helm list -n monitoring | grep -q "prometheus-operator"; then
    echo "Upgrading existing Prometheus Operator installation..."
    helm upgrade prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        -f prometheus-values.yaml
else
    echo "Installing new Prometheus Operator..."
    helm install prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        -f prometheus-values.yaml
fi

# Wait for Prometheus Operator to be ready
echo "Waiting for Prometheus Operator to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus-operator-kube-p-operator -n monitoring || true

# Apply GCP monitoring configuration
echo "Applying GCP monitoring configuration..."
kubectl apply -f gcp-monitoring.yaml

# Setting up Cloud Operations Dashboard configuration
echo "Setting up Cloud Operations Dashboard configuration..."

# Export Cloud Monitoring dashboard JSON
mkdir -p gcp-dashboards
cat <<EOF > gcp-dashboards/neurashield-dashboard.json
{
  "displayName": "NeuraShield Operations Dashboard",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "height": 4,
        "widget": {
          "title": "API Request Rate",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [
              {
                "minAlignmentPeriod": "60s",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    },
                    "filter": "metric.type=\"custom.googleapis.com/api_requests_total\" resource.type=\"k8s_container\""
                  }
                }
              }
            ]
          }
        },
        "width": 6,
        "xPos": 0,
        "yPos": 0
      },
      {
        "height": 4,
        "widget": {
          "title": "API Latency (p95)",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [
              {
                "minAlignmentPeriod": "60s",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_PERCENTILE_95"
                    },
                    "filter": "metric.type=\"custom.googleapis.com/api_latency\" resource.type=\"k8s_container\""
                  }
                }
              }
            ]
          }
        },
        "width": 6,
        "xPos": 6,
        "yPos": 0
      },
      {
        "height": 4,
        "widget": {
          "title": "Blockchain Transactions",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [
              {
                "minAlignmentPeriod": "60s",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    },
                    "filter": "metric.type=\"custom.googleapis.com/blockchain_transactions_total\" resource.type=\"k8s_container\""
                  }
                }
              }
            ]
          }
        },
        "width": 6,
        "xPos": 0,
        "yPos": 4
      },
      {
        "height": 4,
        "widget": {
          "title": "AI Predictions",
          "xyChart": {
            "chartOptions": {
              "mode": "COLOR"
            },
            "dataSets": [
              {
                "minAlignmentPeriod": "60s",
                "plotType": "LINE",
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    },
                    "filter": "metric.type=\"custom.googleapis.com/ai_predictions_total\" resource.type=\"k8s_container\""
                  }
                }
              }
            ]
          }
        },
        "width": 6,
        "xPos": 6,
        "yPos": 4
      }
    ]
  }
}
EOF

echo -e "${GREEN}Monitoring infrastructure setup complete!${NC}"

# Display access information
echo -e "${YELLOW}Access Information:${NC}"
echo "1. Prometheus UI:"
echo "   kubectl port-forward svc/prometheus-operator-kube-p-prometheus 9090:9090 -n monitoring"
echo "   http://localhost:9090"
echo ""
echo "2. Grafana Dashboard:"
echo "   Username: admin"
echo "   Password: NeuraShield2023!"
echo "   kubectl port-forward svc/prometheus-operator-grafana 3000:80 -n monitoring"
echo "   http://localhost:3000"
echo ""
echo "3. Google Cloud Operations Dashboards:"
echo "   Access via Google Cloud Console > Monitoring > Dashboards"
echo ""
echo "4. For troubleshooting:"
echo "   kubectl logs -f deploy/prometheus-to-sd -n monitoring"
echo "   kubectl logs -f ds/fluentbit-agent -n monitoring" 