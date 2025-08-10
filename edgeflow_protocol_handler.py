# Requires: pip install pycryptodome
import zlib
import json
import os
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from typing import Dict, Any, Optional
import random


def _pad_key(key_bytes: bytes, length: int = 32) -> bytes:
    """Pad or trim key to desired length (for AES-256 use 32 bytes)."""
    if len(key_bytes) >= length:
        return key_bytes[:length]
    return key_bytes.ljust(length, b'\0')


def aes_gcm_encrypt(plaintext: bytes, key: bytes) -> Dict[str, bytes]:
    """
    Encrypt plaintext using AES-GCM.
    Returns dict with: ciphertext, nonce, tag
    """
    nonce = get_random_bytes(12)  # recommended 12 bytes for GCM
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return {"ciphertext": ciphertext, "nonce": nonce, "tag": tag}


def aes_gcm_decrypt(ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def compute_hashes(data: bytes) -> Dict[str, str]:
    """Compute SHA-256 and MD5 hex digests (MD5 included for legacy / display only)."""
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    return {"sha256": sha256, "md5": md5}


def simulate_send_over_cn(name: str, payload: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate sending a named content object over a 'CN' style transport.
    This is a placeholder — in a real system replace this with actual network code.
    Returns a delivery-report-like dict.
    """
    # Example behavior: write to local file, record metadata
    out_dir = "edgeflow_out"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{name}.bin")
    with open(filename, "wb") as f:
        f.write(payload)

    # Return a simple simulated ACK / metadata confirmation
    return {
        "status": "sent",
        "name": name,
        "stored_path": filename,
        "metadata": metadata
    }


def edgeflow_protocol_handler(
    task: Dict[str, Any],
    channel_gain: float,
    protocol: str = "CN",
    aes_key: Optional[bytes] = None,
    compress_level: int = 6
) -> Dict[str, Any]:
    """
    Enhanced EDGEFLOW handler:
    - decides transport mode (TCP-like vs UDP-like) based on task 'criticality'
    - applies priority-based compression and channel-aware adjustment
    - encrypts payload using AES-GCM (authenticated encryption)
    - computes SHA-256 and MD5 digests
    - simulates sending via a protocol (default 'CN' simulation)
    - returns a dict with metadata and send-report

    Parameters:
        task: dict containing at least:
            - 'initial_data_size' (float) in MB or arbitrary units
            - 'payload' (bytes) optional; if not present we create dummy bytes based on size
            - 'priority': 'High'|'Medium'|'Low' (optional)
            - 'criticality': bool (optional)
            - 'name': optional unique name string for content
        channel_gain: float indicating channel quality
        protocol: protocol name string ('CN' by default)
        aes_key: bytes key for AES; if None, a random key is generated (returned in metadata)
        compress_level: 0-9 zlib compression level
    """

    # --- Setup / defaults ---
    priority_map = {"High": 0.8, "Medium": 0.6, "Low": 0.4}
    priority = task.get("priority", "Medium")
    critical = bool(task.get("criticality", False))
    name = task.get("name", f"task_{int(random.random()*1e9)}")

    # 1) Transport mode selection
    transport_mode = "TCP-like (Reliable)" if critical else "UDP-like (Fast)"

    # 2) Base compression ratio from priority
    compression_ratio = priority_map.get(priority, 0.5)

    # 3) Channel-aware adjustment (example threshold)
    #    If channel is good, we can compress more aggressively (smaller payload)
    if channel_gain is not None and channel_gain > 5e-5:
        compression_ratio *= 0.9  # 10% more aggressive compression

    # 4) Prepare payload bytes
    initial_size = float(task.get("initial_data_size", 1.0))  # in MB or arbitrary unit
    # If explicit payload bytes included, use them; else create dummy bytes of given size
    if "payload" in task and isinstance(task["payload"], (bytes, bytearray)):
        payload_plain = bytes(task["payload"])
    else:
        # create synthetic payload with pseudo-random bytes scaled to 'initial_size' MB
        # caution: large sizes may be slow; this is for simulation only
        bytes_per_unit = 1024 * 1024  # treat initial_data_size as MB
        payload_plain = get_random_bytes(max(1, int(initial_size * bytes_per_unit)))

    # 5) Compression (zlib)
    compressed = zlib.compress(payload_plain, level=compress_level)
    compressed_size = len(compressed) / (1024 * 1024)  # MB

    # 6) Encryption using AES-GCM
    #    If user didn't provide key, generate one and include in metadata (for simulation only).
    if aes_key is None:
        aes_key = get_random_bytes(32)  # AES-256
        provided_key = False
    else:
        aes_key = _pad_key(aes_key, 32)
        provided_key = True

    enc = aes_gcm_encrypt(compressed, aes_key)
    ciphertext = enc["ciphertext"]
    nonce = enc["nonce"]
    tag = enc["tag"]
    final_size = len(ciphertext) / (1024 * 1024)  # MB

    # 7) Overhead for encryption / headers (simulate fixed overhead like your example)
    encryption_overhead_mb = 0.5  # you used +0.5 earlier; keep it for comparability
    final_size_with_overhead = final_size + encryption_overhead_mb

    # 8) Compute hashes (on plaintext and on ciphertext)
    plaintext_hashes = compute_hashes(payload_plain)
    ciphertext_hashes = compute_hashes(ciphertext)

    # 9) Prepare metadata
    metadata = {
        "transport_mode": transport_mode,
        "priority": priority,
        "channel_gain": channel_gain,
        "compression_ratio_used": compression_ratio,
        "initial_size_MB": initial_size,
        "compressed_size_MB": compressed_size,
        "ciphertext_size_MB": final_size,
        "final_size_with_encryption_overhead_MB": final_size_with_overhead,
        "plaintext_sha256": plaintext_hashes["sha256"],
        "plaintext_md5": plaintext_hashes["md5"],
        "cipher_sha256": ciphertext_hashes["sha256"],
        "cipher_md5": ciphertext_hashes["md5"],
        "aes_nonce_hex": nonce.hex(),
        "aes_tag_hex": tag.hex(),
        "aes_key_provided": provided_key,
    }

    # If we generated a key for simulation, include it in metadata (DO NOT do this in real systems)
    if not provided_key:
        metadata["simulated_aes_key_hex"] = aes_key.hex()

    # 10) Protocol-specific handling
    send_report = None
    if protocol.upper() == "CN":
        # Example: Content-centric publish of named data
        # Name might incorporate user id, task id, priority, timestamp etc.
        content_name = f"{name}_p{priority}_cg{int(channel_gain * 1e6)}"
        send_report = simulate_send_over_cn(content_name, ciphertext, metadata)
    else:
        # Fallback: generic send (we simulate by writing to file)
        content_name = f"{name}_{protocol}"
        send_report = simulate_send_over_cn(content_name, ciphertext, metadata)

    # 11) Return package
    result = {
        "task_name": name,
        "metadata": metadata,
        "send_report": send_report
    }
    return result