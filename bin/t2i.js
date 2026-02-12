#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

const pythonScript = path.join(__dirname, '../client.py');
const args = process.argv.slice(2);

const child = spawn('python3', [pythonScript, ...args], {
    stdio: 'inherit'
});

child.on('exit', (code) => {
    process.exit(code);
});
