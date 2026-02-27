from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
import os, base64, json
from datetime import datetime

class CryptoVerifier:
    def __init__(self):
        self.sessions = {}  # store active challenges

    # Called at user REGISTRATION
    def generate_keypair(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

        return private_pem, public_pem

    # Called when session STARTS — send this to user
    def generate_challenge(self, user_id: str) -> str:
        challenge = base64.b64encode(os.urandom(32)).decode()
        self.sessions[user_id] = {
            "challenge": challenge,
            "timestamp": datetime.utcnow().isoformat()
        }
        return challenge

    # Called to VERIFY user's signed response
    def verify_signature(self, user_id: str, signature_b64: str, public_pem: str) -> bool:
        try:
            session = self.sessions.get(user_id)
            if not session:
                return False

            challenge = session["challenge"].encode()
            signature = base64.b64decode(signature_b64)

            public_key = serialization.load_pem_public_key(
                public_pem.encode(),
                backend=default_backend()
            )

            public_key.verify(
                signature,
                challenge,
                padding.PKCS1v15(),
                hashes.SHA256()
            )

            del self.sessions[user_id]  # one-time use
            return True

        except Exception:
            return False