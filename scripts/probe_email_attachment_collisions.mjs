// Offline diagnostic of the installed pinned upstream writer. No IMAP/model calls.
import { pathToFileURL } from 'node:url';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

const source = process.argv[2];
if (!source) throw new Error('Pass the verified cached qqmail-mcp src/index.js path.');
const { saveAttachment } = await import(pathToFileURL(path.resolve(source)).href);
const root = path.resolve('.agent/email-collision-probes', randomUUID());
await mkdir(root, { recursive: true });
const input = { baseDir: root, buffer: Buffer.from('FIRST_ONLY'), folder: 'INBOX', uid: 123,
  part: '2', filename: 'report.txt', contentType: 'text/plain', applyQuarantine: async () => false };
const first = await saveAttachment(input);
const failures = {};
for (const [label, patch] of [
  ['same_attachment_retry', {}],
  ['different_part_same_filename', { part: '3', buffer: Buffer.from('SECOND_ONLY') }],
]) {
  try {
    const saved = await saveAttachment({ ...input, ...patch });
    failures[label] = { status: 'SAVED', distinct_path: saved.path !== first.path };
  }
  catch (error) { failures[label] = error.message; }
}
const result = { root, failures, original_preserved: (await readFile(first.path, 'utf8')) === 'FIRST_ONLY',
  note: 'No account/UIDVALIDITY argument in this writer; cross-account risk is source-derived, not a live account-switch test.' };
await writeFile(path.join(root, 'summary.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result));
