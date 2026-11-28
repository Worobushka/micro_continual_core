#!/usr/bin/env node
// Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov & Sofia Sidorenko
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// Cross-architecture determinism runner for Suite C (Q8.24/Q39.24) on wasm32-wasip1 via Node.js WASI.
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
import path from 'node:path';
import process from 'node:process';

const wasmPath = process.argv[2]
  || path.resolve('target/wasm32-wasip1/release/bench_step8_v050.wasm');
const args = ['bench_step8_v050.wasm', ...(process.argv.slice(3).length ? process.argv.slice(3) : ['--suite', 'c'])];

const wasi = new WASI({
  version: 'preview1',
  args,
  env: process.env,
  preopens: {
    '.': process.cwd(),
    '..': path.resolve('..'),
  },
});

const wasmBytes = await readFile(wasmPath);
const wasmModule = await WebAssembly.compile(wasmBytes);
const instance = await WebAssembly.instantiate(wasmModule, wasi.getImportObject());
wasi.start(instance);
