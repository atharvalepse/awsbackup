#!/usr/bin/env bash
# Provision the GML VM on Google Cloud Compute Engine.
#
# Run from your laptop after `gcloud auth login` and `gcloud config set project <ID>`.
# Edit the variables at top, then: bash 01-create-vm.sh
#
# What this creates:
#   * 1× e2-medium VM (2 vCPU, 4 GB RAM)
#   * 100 GB persistent SSD ("standard" disk is cheaper but slower for Postgres)
#   * Firewall rules: allow 80, 443 from anywhere; allow 22 from your IP only
#   * Static external IP (so DNS doesn't break on restart)
#
# Cost on the $300 free trial:
#   e2-medium  ~$25/mo
#   100 GB SSD ~$17/mo
#   static IP  ~$3/mo  (only billed when attached)
#   ──────────────────
#   ~$45/mo → ~6–7 months of runway on the credit.

set -euo pipefail

# ============================ EDIT THESE ============================
PROJECT_ID="${GCP_PROJECT_ID:-gml-prod}"   # gcloud project id
REGION="${GCP_REGION:-us-central1}"        # pick closest to your users
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-gml-prod}"
DISK_SIZE_GB="${DISK_SIZE_GB:-100}"        # raise if you expect >50 users
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"  # 2 vCPU / 4 GB RAM
SSH_SOURCE_CIDR="${SSH_SOURCE_CIDR:-0.0.0.0/0}"  # tighten to YOUR_IP/32
# ====================================================================

echo "▶ Setting active project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

echo "▶ Enabling required APIs (idempotent)..."
gcloud services enable compute.googleapis.com storage.googleapis.com

echo "▶ Reserving static external IP: ${VM_NAME}-ip"
gcloud compute addresses create "${VM_NAME}-ip" \
  --region="$REGION" \
  || echo "  (already exists)"
EXT_IP=$(gcloud compute addresses describe "${VM_NAME}-ip" --region="$REGION" --format='value(address)')
echo "  → $EXT_IP"

echo "▶ Creating VM: $VM_NAME ($MACHINE_TYPE in $ZONE)"
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="debian-12" \
  --image-project="debian-cloud" \
  --boot-disk-size="${DISK_SIZE_GB}GB" \
  --boot-disk-type="pd-ssd" \
  --tags="http-server,https-server,gml-api" \
  --address="$EXT_IP" \
  --metadata=enable-oslogin=TRUE \
  --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring \
  || echo "  (already exists)"

echo "▶ Firewall rules"
# HTTP/HTTPS from everywhere — nginx will TLS + auth
gcloud compute firewall-rules create allow-http-https \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:80,tcp:443 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server,https-server \
  || echo "  (http-https rule already exists)"

# SSH only from your IP (tighten this!)
gcloud compute firewall-rules create allow-ssh-restricted \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges="$SSH_SOURCE_CIDR" \
  || echo "  (ssh rule already exists)"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  VM ready."
echo "  External IP:   $EXT_IP"
echo "  Zone:          $ZONE"
echo ""
echo "  Next steps:"
echo "    1. Point your DNS A record at $EXT_IP"
echo "    2. SSH in:    gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "    3. On the VM, run:  bash 02-install-postgres.sh"
echo "═══════════════════════════════════════════════════════════════"
