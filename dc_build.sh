echo "🧩    Updating repository..."
echo
git pull --ff-only

echo 🛑 Stopping containers...
docker-compose down
echo

echo 🧩 Rebuilding images...
docker-compose build
echo

echo 🚀 Starting containers...
docker-compose up -d
echo


echo "frontend at http://localhost:8080"
echo
echo "API at http://localhost:5002"

echo

echo 📜 Showing logs...
docker-compose logs -f
