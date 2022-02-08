# -*- coding: utf-8 -*-
"""socket_communication.py - Handles communications with the PSX PSU.

Todo:
    * Figure out how this works and document it properly.
"""

# region Import modules.
from __future__ import annotations
import logging
import socket
import select
from contextlib import contextmanager
# endregion


class InstrumentError(Exception):
    pass


def ensure_bytes(x):
    if isinstance(x, bytes):
        return x
    else:
        return str(x).encode("ascii")


def ensure_str(x):
    if isinstance(x, bytes):
        return x.decode("ascii")
    else:
        return str(x)


class InstrumentSocket:
    def __init__(self, ipaddr, socketnr, *, timeout=2000,
                 read_termination="\n", verbose=False, write_termination="\n"):
        self.rsrc_name = None
        self.portnr = socketnr
        self.ipaddr = ipaddr
        self.verbose = verbose
        self.read_termination = read_termination
        self.write_termination = write_termination
        self.timeout = timeout
        self.resource = None
        self.logger = logging.getLogger(__name__)
        self.debug = self.logger.debug

    def __init_subclass__(cls):
        _drivers[cls.__name__.lower()] = cls

    @contextmanager
    def open_instrument(self):
        try:
            resource = self.instrument_open()
            yield resource
        finally:
            self.instrument_close()

    def instrument_open(self):
        if self.resource is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 
                                     socket.IPPROTO_TCP)
                sock.settimeout(self.timeout / 1000)
                sock.connect((self.ipaddr, self.portnr))
                self.resource = sock
            except socket.gaierror as exc:
                raise InstrumentError(
                    f"Tried to open a {self.__class__}" +
                    f" using resource: {self.rsrc_name}",
                    exc.error_code) from exc
            self.resource = sock
        return self.resource

    def instrument_close(self):
        self.resource.close()
        self.resource = None

    def write(self, cmd):
        termination = self.write_termination
        
        if termination is not None and not cmd.endswith(termination):
            cmd += termination
        if self.resource is None:
            raise InstrumentError(
                "You must open a connection to the PSX first")
        if self.verbose:
            self.debug(f"{self.__class__.__name__}.write: {cmd!r}")
        self.resource.send(ensure_bytes(cmd))

    def read(self, n=10000, strip=True):
        if self.resource is None:
            raise InstrumentError(
                "You must open a connection to the PSX first")
        out = []
        while True:
            self.resource.setblocking(0)
            ready = select.select([self.resource], [], [], 10)
            if ready[0]:
                out.append(ensure_str(self.resource.recv(n)))
                if out[-1][-1:] == "\n":
                    break
            else:
                return 'timeout'

        out = "".join(out)
        if len(out) > 1:
            output = out.split("\n", 1)
            if len(output) > 1 and len(output[1]) > 0:
                print(f"WARNING, read got data after EOL marker {out!r}")
            if strip:
                out = output[0].strip()
            else:
                out = output[0]
        if self.verbose:
            self.debug(f"{self.__class__.__name__}.read: {out[:80]!r}...")

        return out

    def query(self, cmd, n=10000, strip=True):
        self.write(cmd)
        x = self.read(n)
        if x == 'timeout':
            self.write(cmd)
            x = self.read(n)
        return x.strip()

    def query_str(self, cmd, n=10000, strip=True):
        res = self.query(cmd, n, strip=strip)
        return ensure_str(res)
