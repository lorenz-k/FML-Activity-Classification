#!/bin/bash
set -e

PROJECT=fml-ac
REGION=europe-west3
ZONE=europe-west3-a
REPO=europe-west3-docker.pkg.dev/fml-ac/fml-ac
IMAGE=$REPO/fml-flower:latest
SERVER_IP=10.156.0.2

RUN_ID=${RUN_ID:-gcp_run_$(date +%Y%m%d_%H%M%S)}
ROUNDS=${ROUNDS:-5}
NUM_CLIENTS=${NUM_CLIENTS:-4}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.001}
HIDDEN_DIM=${HIDDEN_DIM:-128}
DROPOUT=${DROPOUT:-0.2}

echo "=== FML Deployment ==="
echo "Image:   $IMAGE"
echo "Run ID:  $RUN_ID"
echo "Rounds:  $ROUNDS"
echo ""

# 1. Push image
echo "[1/4] Pushing image to Artifact Registry..."
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
sudo docker tag fml-flower:local $IMAGE
sudo docker push $IMAGE
echo "Done."

# 2. Start VMs
echo "[2/4] Starting VMs..."
gcloud compute instances start fl-server fl-client-0 fl-client-1 fl-client-2 fl-client-3 \
  --zone=$ZONE --project=$PROJECT
echo "Waiting 30s for VMs to boot..."
sleep 30

# 3. Start server
echo "[3/4] Starting server on fl-server..."
gcloud compute ssh fl-server --zone=$ZONE --project=$PROJECT --command="
  sudo docker stop fl-server 2>/dev/null || true
  sudo docker rm fl-server 2>/dev/null || true
  gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin $REGION-docker.pkg.dev
  sudo docker pull $IMAGE
  sudo docker run -d --name fl-server \
    -p 8080:8080 \
    -e RUN_ID=$RUN_ID \
    -e ROUNDS=$ROUNDS \
    -e NUM_CLIENTS=$NUM_CLIENTS \
    -e LOCAL_EPOCHS=$LOCAL_EPOCHS \
    -e BATCH_SIZE=$BATCH_SIZE \
    -e LR=$LR \
    -e HIDDEN_DIM=$HIDDEN_DIM \
    -e DROPOUT=$DROPOUT \
    $IMAGE \
    python -m src.flower_server --server-address 0.0.0.0:8080
  echo 'Server started.'
"

echo "Waiting 10s for server to be ready..."
sleep 10

# 4. Start clients
echo "[4/4] Starting clients..."
for i in 0 1 2 3; do
  echo "  Starting fl-client-$i..."
  gcloud compute ssh fl-client-$i --zone=$ZONE --project=$PROJECT --command="
    sudo docker stop fl-client 2>/dev/null || true
    sudo docker rm fl-client 2>/dev/null || true
    gcloud auth print-access-token | sudo docker login -u oauth2accesstoken --password-stdin $REGION-docker.pkg.dev
    sudo docker pull $IMAGE
    sudo docker run -d --name fl-client \
      -e CLIENT_ID=$i \
      -e SERVER_ADDRESS=$SERVER_IP:8080 \
      -e LOCAL_EPOCHS=$LOCAL_EPOCHS \
      -e BATCH_SIZE=$BATCH_SIZE \
      -e LR=$LR \
      -e HIDDEN_DIM=$HIDDEN_DIM \
      -e DROPOUT=$DROPOUT \
      $IMAGE \
      python -m src.flower_client
    echo 'Client $i started.'
  " &
done
wait


echo "Waiting for federated training to finish..."

gcloud compute ssh fl-server \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --command="
    sudo docker wait fl-server
    sudo docker logs fl-server
  "

echo ""
echo "=== Deployment complete ==="
echo "Logs: gcloud compute ssh fl-server --zone=$ZONE --command='sudo docker logs -f fl-server'"
