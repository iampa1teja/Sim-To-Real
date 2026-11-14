"""Loopback packet accounting (Ethernet/IP/TCP overhead and ACKs included).

Run in the dedicated benchmark network namespace as root/CAP_NET_RAW. Payloads
are snaplen-truncated; packet lengths and headers suffice for accounting.
"""
import csv
import gzip
from pathlib import Path
import re
import signal
import struct
import subprocess
import time

class Capture:
    def __init__(self, destination, port):
        self.path = Path(destination).with_suffix('.pcap')
        self.port = port
        self.process = subprocess.Popen(['tcpdump', '--immediate-mode', '-i', 'lo', '-n', '-U', '-B', '32768', '-s', '96',
            '-w', str(self.path), 'tcp', 'port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        line = self.process.stderr.readline()
        while b'listening on' not in line:
            if not line:
                raise RuntimeError('tcpdump failed to start')
            line = self.process.stderr.readline()

    def stop(self, iterations, *, require_exchange=True):
        # Give the capture consumer time to drain already delivered kernel traffic.
        time.sleep(.15)
        self.process.send_signal(signal.SIGINT)
        _, error = self.process.communicate(timeout=10)
        drops = re.search(rb'(\d+) packets dropped by kernel', error)
        if self.process.returncode != 0 or drops is None or int(drops[1]) != 0:
            raise RuntimeError(f'Unreliable packet capture: {error!r}')
        counts = {'request': {'packets': 0, 'l2_bytes': 0, 'ip_bytes': 0, 'tcp_payload_bytes': 0},
                  'response': {'packets': 0, 'l2_bytes': 0, 'ip_bytes': 0, 'tcp_payload_bytes': 0}}
        with self.path.open('rb') as stream, gzip.open(self.path.with_suffix('.packets.csv.gz'), 'wt') as output:
            writer = csv.writer(output)
            writer.writerow(['seconds', 'microseconds', 'src_port', 'dst_port', 'l2_bytes', 'ip_bytes', 'tcp_payload_bytes'])
            header = stream.read(24)
            endian = '<' if header[:4] == b'\xd4\xc3\xb2\xa1' else '>'
            assert struct.unpack(endian + 'I', header[20:24])[0] == 1, 'Expected Ethernet loopback capture'
            while record := stream.read(16):
                sec, usec, captured, length = struct.unpack(endian + 'IIII', record)
                packet = stream.read(captured)
                if packet[12:14] != b'\x08\x00':
                    continue
                ip_offset = 14
                ip_header = (packet[ip_offset] & 15) * 4
                ip_length = int.from_bytes(packet[ip_offset + 2:ip_offset + 4], 'big')
                tcp_offset = ip_offset + ip_header
                src, dst = struct.unpack('!HH', packet[tcp_offset:tcp_offset + 4])
                tcp_header = (packet[tcp_offset + 12] >> 4) * 4
                tcp_payload = ip_length - ip_header - tcp_header
                direction = 'request' if dst == self.port else 'response'
                row = counts[direction]
                row['packets'] += 1
                row['l2_bytes'] += length
                row['ip_bytes'] += ip_length
                row['tcp_payload_bytes'] += tcp_payload
                writer.writerow([sec, usec, src, dst, length, ip_length, tcp_payload])
        if require_exchange:
            assert counts['request']['tcp_payload_bytes'] > 0 and counts['response']['tcp_payload_bytes'] > 0, 'Empty wire capture'
        self.path.unlink()
        return {'kernel_dropped_packets': int(drops[1]), 'tcpdump_summary': error.decode(),
                'totals': counts,
                'per_exchange': {direction: {key: value / iterations for key, value in row.items()}
                                 for direction, row in counts.items()},
                'note': 'Loopback packet accounting; includes TCP ACKs and protocol framing, excludes admission/warmup. '
                        'No physical Ethernet/FCS/preamble on lo; GRO/GSO affect packet segmentation.'}
