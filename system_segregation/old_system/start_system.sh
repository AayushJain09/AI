#!/bin/bash
# Production System Startup Script

echo "🚀 Starting AI Recognition System"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "✅ Activating virtual environment"
    source venv/bin/activate
fi

# Check system status
echo "📊 System Status Check:"
echo "- Index file: $(ls -lh data/models/faiss_index.bin 2>/dev/null | awk '{print $5}' || echo 'NOT FOUND')"
echo "- Config file: $([ -f config.yaml ] && echo 'OK' || echo 'MISSING')"
echo "- Models available: $(ls checkpoints/ | wc -l) files"

# Test recognition system
echo ""
echo "🔍 Testing Recognition System..."
python3 test_recognition_final.py
if [ $? -eq 0 ]; then
    echo "✅ Recognition system test PASSED"
else
    echo "❌ Recognition system test FAILED"
    exit 1
fi

echo ""
echo "🌐 Starting Backend Server..."
python3 backend/main.py &
BACKEND_PID=$!
sleep 5

echo "🖥️  Starting Frontend Application..."
python3 frontend/main.py &
FRONTEND_PID=$!

echo ""
echo "✅ System started successfully!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop the system"

# Wait for interrupt
trap "echo '🛑 Stopping system...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT
wait