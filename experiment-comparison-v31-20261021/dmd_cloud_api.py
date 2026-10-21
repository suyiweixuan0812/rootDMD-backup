import configparser
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/tmp/dmd-volc-sdk')
import requests
from volcengine.auth.SignerV4 import SignerV4
from volcengine.Credentials import Credentials
from volcengine.base.Request import Request

STATE = Path('/tmp/dmd-cloud-state')
STATE.mkdir(mode=0o700, exist_ok=True)


def call(action, body=None, version='2024-07-01', service='ml_platform'):
    config = configparser.ConfigParser()
    config.read('/root/.volc/credentials')
    credentials = Credentials(config['default']['access_key_id'],
                              config['default']['secret_access_key'],
                              service, 'cn-beijing')
    request = Request()
    request.schema = 'https'
    request.method = 'POST'
    request.host = 'open.volcengineapi.com'
    request.path = '/'
    request.query = dict(Action=action, Version=version)
    request.headers = {'Host': request.host, 'Content-Type': 'application/json'}
    request.body = json.dumps(body or {}, separators=(',', ':'), ensure_ascii=False)
    SignerV4.sign(request, credentials)
    client = requests.Session()
    client.trust_env = False
    response = client.post(request.build(), headers=request.headers,
                             data=request.body.encode('utf-8'), timeout=(10, 40))
    data = response.json()
    target = STATE / (action + '.json')
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, indent=2)
    return data


def scrub(value):
    if isinstance(value, dict):
        return {key: ('[REDACTED]' if any(word in key.lower() for word in
                    ['secret', 'password', 'token', 'credential', 'key', 'envs', 'entrypoint'])
                    else scrub(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


if __name__ == '__main__':
    print(json.dumps(scrub(call(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})),
                     ensure_ascii=False))
