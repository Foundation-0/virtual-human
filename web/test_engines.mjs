// Checks the JavaScript engines against the Python engines' exported references.
//   node web/test_engines.mjs
import { createRequire } from "module";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const MD = require(join(here, "vh-md.js"));
const PK = require(join(here, "vh-pbpk.js"));
let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`); if (!ok) failures++; };

// --- molecular engine vs JAX autodiff -----------------------------------------
for (const key of ["chignolin", "trpcage"]) {
  const sys = JSON.parse(readFileSync(join(here, "data", key + ".json")));
  const eng = new MD.Engine(sys);
  const x = Float64Array.from(sys.reference.x.flat());
  const E = eng.compute(x);
  const relE = Math.abs(E - sys.reference.energy) / Math.abs(sys.reference.energy);
  let maxTerm = 0;
  for (const [k, v] of Object.entries(sys.reference.terms)) maxTerm = Math.max(maxTerm, Math.abs(eng.terms[k] - v) / Math.max(Math.abs(v), 1));
  const fr = sys.reference.forces.flat();
  let num = 0, den = 0, maxAbs = 0;
  for (let i = 0; i < fr.length; i++) { const d = eng.f[i] - fr[i]; num += d * d; den += fr[i] * fr[i]; maxAbs = Math.max(maxAbs, Math.abs(d)); }
  const relF = Math.sqrt(num / den);
  check(relE < 1e-10 && maxTerm < 1e-9, `${key}: energy rel err ${relE.toExponential(1)}, worst term ${maxTerm.toExponential(1)}`);
  check(relF < 1e-9, `${key}: force rel RMS err ${relF.toExponential(1)} (max abs ${maxAbs.toExponential(1)} kJ/mol/nm)`);
  // speed + short dynamics sanity
  const dyn = new MD.Dynamics(eng, sys.x0, { temperature: 300, seed: 3 });
  const t0 = performance.now(); dyn.step(500); const ms = (performance.now() - t0) / 500;
  check(isFinite(dyn.e.energy) && dyn.temperature() > 150 && dyn.temperature() < 450,
    `${key}: 500 steps ok, T=${dyn.temperature().toFixed(0)} K, ${ms.toFixed(2)} ms/step -> ${(1 / ms * 1000 * 0.001).toFixed(2)} ps per wall second`);
}

// --- body engine vs Python ------------------------------------------------------
const body = JSON.parse(readFileSync(join(here, "data", "body.json")));
const ref = body.reference_profile;
const model = new PK.Model(body, ref.individual);
const pt = new PK.Patient(model);
let worst = 0, k = 0;
const doses = ref.doses.map(d => ({ t: d[0], mg: d[1], route: d[2], dur: d[3] || 0 }));
for (let i = 0; i < ref.time_h.length; i++) {
  const t = ref.time_h[i];
  for (const d of doses) if (Math.abs(d.t - pt.t) < 1e-9 && !d.done) { pt.dose(d.mg, d.route, d.dur); d.done = true; }
  if (i > 0) pt.advance(t - ref.time_h[i - 1]);
  for (const d of doses) if (Math.abs(d.t - pt.t) < 1e-9 && !d.done) { pt.dose(d.mg, d.route, d.dur); d.done = true; }
  const o = pt.observe();
  if (ref.plasma[i] > 1e-6) { worst = Math.max(worst, Math.abs(o.plasma - ref.plasma[i]) / ref.plasma[i]); k++; }
}
check(worst < 1e-6, `body: JS vs Python plasma profile, worst rel err ${worst.toExponential(1)} over ${k} points`);
check(Math.abs(pt.observe().total - 9.5) < 1e-9, `body: mass conserved (${pt.observe().total.toFixed(12)} mg of 9.5 mg)`);

if (failures) { console.log(`\n${failures} check(s) failed`); process.exit(1); }
console.log("\nall JS engine checks passed");
