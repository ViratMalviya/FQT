1️⃣ System Requirements (ALL PCs)
Hardware

3 PCs on the same Wi-Fi / LAN

1 → Server

2 → Clients

Software

OS: Windows / Linux

Python 3.9 – 3.11

Minimum RAM: 8 GB recommended

2️⃣ Project Folder Structure (ALL PCs)

Create a folder named:

federated_qcnn/


Inside it, create:

federated_qcnn/
│
├── server.py
├── client.py
├── model.py
├── data_utils.py
├── requirements.txt
└── dataset/
    ├── REAL/
    └── FAKE/


⚠️ Important

Each client PC must have its own dataset inside dataset/

Data must NOT be shared between PCs

3️⃣ Dataset Preparation (CLIENT PCs ONLY)

On each client PC:

dataset/
├── REAL/   → real audio (.wav)
└── FAKE/   → fake audio (.wav)


✔ You can split the dataset manually
✔ Files can be different across clients
✔ Server does NOT need the dataset

4️⃣ Create Virtual Environment (ALL PCs)

Open terminal inside federated_qcnn/

Create venv
python -m venv venv

Activate venv

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate


You should see (venv) in terminal.

5️⃣ Install Dependencies (ALL PCs)
requirements.txt
torch
torchvision
torchaudio
pennylane
librosa
numpy
scikit-learn
flwr


Install:

pip install -r requirements.txt


If librosa errors appear:

pip install soundfile

6️⃣ Identify Server IP Address (SERVER PC)

On Server PC only:

Windows
ipconfig

Linux
ifconfig


Example output:

IPv4 Address: 192.168.1.10


📌 Note this IP address
You will paste it into client.py on both client PCs.

7️⃣ Update Server IP in Clients (CLIENT PCs)

Open client.py on each client PC.

Find:

server_address="192.168.1.10:8080"


Replace 192.168.1.10 with your actual server IP.

8️⃣ Firewall Configuration (IMPORTANT)
On Server PC

Allow port 8080.

Windows

Open Windows Defender Firewall

Allow inbound TCP on port 8080

Linux
sudo ufw allow 8080

9️⃣ How to Run the System (ORDER MATTERS)
STEP 1 — Start Server (SERVER PC)
python server.py


Expected output:

Federated server running on 0.0.0.0:8080


⚠️ Do NOT close this terminal

STEP 2 — Start Client 1 (CLIENT PC 1)
python client.py


Expected:

Client connected to server

STEP 3 — Start Client 2 (CLIENT PC 2)
python client.py


Expected:

Client connected to server

🔄 What Happens Internally

Each client:

Loads local audio

Extracts MFCC features

Trains quantum-generated CNN parameters

Server:

Receives only model updates

Performs Federated Averaging (FedAvg)

No audio data is shared ❌

Only parameters are shared ✅

10️⃣ How to Confirm Federated Learning Is Working

✔ Server shows training rounds
✔ Clients connect successfully
✔ If one client stops → training fails
✔ Different data on each client → still works

This confirms real federated learning, not simulation.