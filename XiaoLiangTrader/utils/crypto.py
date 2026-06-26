"""
API Key 加密工具 — 用 Fernet (AES) 保护敏感信息
================================================
宿舍电脑也可能被室友借用，密码不能明文存！
"""

import os
from pathlib import Path
from cryptography.fernet import Fernet

# 密钥存放位置：优先环境变量，其次 ~/.xlt/secret.key
_SECRET_ENV = "XLT_SECRET"
_SECRET_FILE = Path.home() / ".xlt" / "secret.key"


def _get_or_create_key() -> bytes:
    """获取或生成 Fernet 密钥"""
    # 1. 环境变量
    env = os.environ.get(_SECRET_ENV)
    if env:
        return env.encode()

    # 2. 本地密钥文件
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_bytes().strip()

    # 3. 首次运行：自动生成
    key = Fernet.generate_key()
    _SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SECRET_FILE.write_bytes(key)
    try:
        os.chmod(_SECRET_FILE, 0o600)  # 仅 owner 可读写
    except OSError:
        pass  # Windows 不支持 Unix 权限
    print(f"[crypto] 首次运行，密钥已生成: {_SECRET_FILE}")
    return key


def _fernet() -> Fernet:
    return Fernet(_get_or_create_key())


def encrypt(plaintext: str) -> str:
    """加密字符串，返回 base64 编码的密文"""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """解密 base64 编码的密文"""
    return _fernet().decrypt(ciphertext.encode()).decode()


def maybe_decrypt(val: str) -> str:
    """
    如果值以 'ENC:' 开头则解密，否则原样返回。
    用于读取 YAML 配置时自动解密敏感字段。
    """
    if isinstance(val, str) and val.startswith("ENC:"):
        return decrypt(val[4:])
    return val


def encrypt_yaml_value(val: str) -> str:
    """加密一个值并加上 'ENC:' 前缀"""
    if val and not val.startswith("ENC:"):
        return "ENC:" + encrypt(val)
    return val


if __name__ == "__main__":
    # 快速测试
    secret = "my_super_secret_api_key"
    enc = encrypt(secret)
    dec = decrypt(enc)
    print(f"原文: {secret}")
    print(f"密文: {enc}")
    print(f"解密: {dec}")
    assert dec == secret
    print("✓ 加密解密正常")
