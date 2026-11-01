/* Virtual Human — real-time molecular engine (browser / Node).
 *
 * A JavaScript port of vh.molecular: AMBER ff14SB bonded terms, Lennard-Jones
 * and Coulomb with 1-4 exceptions, GB-OBC2 implicit water + ACE nonpolar term,
 * Langevin BAOAB dynamics. Forces are analytic (hand-derived, including the
 * chain rule through the Born radii) and are tested against the JAX engine's
 * autodiff forces in web/test_engines.mjs.
 *
 * Units: nm, ps, amu, kJ/mol, e.
 */
(function (root) {
  "use strict";
  const ONE_4PI_EPS0 = 138.93545764438198;
  const KB = 0.00831446261815324;
  const OBC_OFFSET = 0.009, ALPHA = 1.0, BETA = 0.8, GAMMA = 4.85, PROBE = 0.14;

  function Engine(sys) {
    const N = sys.masses.length;
    this.N = N;
    this.sys = sys;
    this.mass = Float64Array.from(sys.masses);
    this.q = Float64Array.from(sys.charges);
    // bonded
    this.bonds = Int32Array.from(sys.bonds.flat());
    this.bk = Float64Array.from(sys.bond_k); this.br0 = Float64Array.from(sys.bond_r0);
    this.angles = Int32Array.from(sys.angles.flat());
    this.ak = Float64Array.from(sys.angle_k); this.at0 = Float64Array.from(sys.angle_t0);
    this.tors = Int32Array.from(sys.torsions.flat());
    this.tk = Float64Array.from(sys.torsion_k); this.tn = Float64Array.from(sys.torsion_n);
    this.tph = Float64Array.from(sys.torsion_phase);
    // nonbonded pair list (exclusions removed), parameters precomputed
    const excl = new Uint8Array(N * N);
    const exc = sys.exceptions;
    const e14 = [];
    for (let m = 0; m < exc.length; m++) {
      const i = exc[m][0], j = exc[m][1];
      excl[i * N + j] = excl[j * N + i] = 1;
      if (sys.exc_qq[m] !== 0 || sys.exc_eps[m] !== 0) e14.push(m);
    }
    const pi = [], pj = [], eps4 = [], sig2 = [], kqq = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      if (excl[i * N + j]) continue;
      pi.push(i); pj.push(j);
      eps4.push(4 * Math.sqrt(sys.epsilon[i] * sys.epsilon[j]));
      const s = 0.5 * (sys.sigma[i] + sys.sigma[j]); sig2.push(s * s);
      kqq.push(ONE_4PI_EPS0 * sys.charges[i] * sys.charges[j]);
    }
    this.pi = Int32Array.from(pi); this.pj = Int32Array.from(pj);
    this.eps4 = Float64Array.from(eps4); this.sig2 = Float64Array.from(sig2); this.kqq = Float64Array.from(kqq);
    this.ei = Int32Array.from(e14.map(m => exc[m][0])); this.ej = Int32Array.from(e14.map(m => exc[m][1]));
    this.ekqq = Float64Array.from(e14.map(m => ONE_4PI_EPS0 * sys.exc_qq[m]));
    this.eeps4 = Float64Array.from(e14.map(m => 4 * sys.exc_eps[m]));
    this.esig2 = Float64Array.from(e14.map(m => sys.exc_sigma[m] * sys.exc_sigma[m]));
    // GB
    this.gb = !!sys.gb_radius;
    if (this.gb) {
      this.rad = Float64Array.from(sys.gb_radius);
      this.or = this.rad.map(r => r - OBC_OFFSET);
      this.sr = Float64Array.from(sys.gb_scale.map((s, i) => s * this.or[i]));
      this.gbPre = -sys.gb_coulomb_constant * (1 / sys.solute_dielectric - 1 / sys.solvent_dielectric);
      this.ace = this.rad.map(r => 4 * Math.PI * sys.surface_area_energy * (r + PROBE) * (r + PROBE));
      this.engulfed = !!sys.gb_engulfed_correction;
      this.I = new Float64Array(N); this.R = new Float64Array(N);
      this.dRdpsi = new Float64Array(N); this.dEdR = new Float64Array(N);
      // dense pair tables (all i<j) so LJ/Coulomb ride along in the GB loop
      const P = N * (N - 1) / 2;
      this.pa = new Int32Array(P); this.pb = new Int32Array(P);
      this.peps4 = new Float64Array(P); this.psig2 = new Float64Array(P); this.pkqq = new Float64Array(P);
      this.pr = new Float64Array(P); this.pdIab = new Float64Array(P); this.pdIba = new Float64Array(P);
      let p = 0;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++, p++) {
        this.pa[p] = i; this.pb[p] = j;
        if (excl[i * N + j]) continue;
        this.peps4[p] = 4 * Math.sqrt(sys.epsilon[i] * sys.epsilon[j]);
        const sg = 0.5 * (sys.sigma[i] + sys.sigma[j]); this.psig2[p] = sg * sg;
        this.pkqq[p] = ONE_4PI_EPS0 * sys.charges[i] * sys.charges[j];
      }
    }
    this.f = new Float64Array(3 * N);
    this.terms = {};
  }

  // HCT descreening integral (with the 0.5 prefactor) and its r-derivative, for atom i
  // (offset radius ori) descreened by a sphere of scaled radius s at distance r.
  function hct(r, ori, s, out) {
    const U = r + s;
    if (!(ori < U)) { out[0] = 0; out[1] = 0; return; }
    const D = Math.abs(r - s);
    let L, dL;
    if (ori > D) { L = ori; dL = 0; } else { L = D; dL = r >= s ? 1 : -1; }
    const u = 1 / U, l = 1 / L, u2 = u * u, l2 = l * l, lnr = Math.log(L / U);
    const T = l - u + 0.25 * (r - s * s / r) * (u2 - l2) + 0.5 * lnr / r;
    const du = -u2, dl = -l2 * dL;
    const dT = dl - du + 0.25 * (1 + s * s / (r * r)) * (u2 - l2)
      + 0.25 * (r - s * s / r) * (2 * u * du - 2 * l * dl)
      + 0.5 * ((dL / L - 1 / U) / r - lnr / (r * r));
    out[0] = 0.5 * T; out[1] = 0.5 * dT;
  }

  Engine.prototype.compute = function (x) {
    const N = this.N, f = this.f;
    f.fill(0);
    let eb = 0, ea = 0, et = 0, elj = 0, ec = 0, elj14 = 0, ec14 = 0, egb = 0, esa = 0;
    // ---- bonds
    const B = this.bonds;
    for (let m = 0, nb = this.bk.length; m < nb; m++) {
      const i = 3 * B[2 * m], j = 3 * B[2 * m + 1];
      const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz), d = r - this.br0[m];
      eb += 0.5 * this.bk[m] * d * d;
      const g = -this.bk[m] * d / r;
      f[i] += g * dx; f[i + 1] += g * dy; f[i + 2] += g * dz;
      f[j] -= g * dx; f[j + 1] -= g * dy; f[j + 2] -= g * dz;
    }
    // ---- angles
    const A = this.angles;
    for (let m = 0, na = this.ak.length; m < na; m++) {
      const i = 3 * A[3 * m], j = 3 * A[3 * m + 1], k = 3 * A[3 * m + 2];
      const ux = x[i] - x[j], uy = x[i + 1] - x[j + 1], uz = x[i + 2] - x[j + 2];
      const vx = x[k] - x[j], vy = x[k + 1] - x[j + 1], vz = x[k + 2] - x[j + 2];
      const ru = Math.sqrt(ux * ux + uy * uy + uz * uz), rv = Math.sqrt(vx * vx + vy * vy + vz * vz);
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      const sn = Math.sqrt(cx * cx + cy * cy + cz * cz), cs = ux * vx + uy * vy + uz * vz;
      const th = Math.atan2(sn, cs), d = th - this.at0[m];
      ea += 0.5 * this.ak[m] * d * d;
      const dEdth = this.ak[m] * d;
      const st = Math.max(sn / (ru * rv), 1e-8), ct = cs / (ru * rv);
      // dtheta/dri = (ct*u^ - v^)/(ru*st); dtheta/drk = (ct*v^ - u^)/(rv*st)
      const gi = -dEdth / (ru * st), gk = -dEdth / (rv * st);
      const fix = gi * (ct * ux / ru - vx / rv), fiy = gi * (ct * uy / ru - vy / rv), fiz = gi * (ct * uz / ru - vz / rv);
      const fkx = gk * (ct * vx / rv - ux / ru), fky = gk * (ct * vy / rv - uy / ru), fkz = gk * (ct * vz / rv - uz / ru);
      f[i] += fix; f[i + 1] += fiy; f[i + 2] += fiz;
      f[k] += fkx; f[k + 1] += fky; f[k + 2] += fkz;
      f[j] -= fix + fkx; f[j + 1] -= fiy + fky; f[j + 2] -= fiz + fkz;
    }
    // ---- torsions (Blondel & Karplus 1996 gradient)
    const T = this.tors;
    for (let m = 0, nt = this.tk.length; m < nt; m++) {
      const i = 3 * T[4 * m], j = 3 * T[4 * m + 1], k = 3 * T[4 * m + 2], l = 3 * T[4 * m + 3];
      const Fx = x[i] - x[j], Fy = x[i + 1] - x[j + 1], Fz = x[i + 2] - x[j + 2];
      const Gx = x[j] - x[k], Gy = x[j + 1] - x[k + 1], Gz = x[j + 2] - x[k + 2];
      const Hx = x[l] - x[k], Hy = x[l + 1] - x[k + 1], Hz = x[l + 2] - x[k + 2];
      const Ax = Fy * Gz - Fz * Gy, Ay = Fz * Gx - Fx * Gz, Az = Fx * Gy - Fy * Gx;
      const Bx = Hy * Gz - Hz * Gy, By = Hz * Gx - Hx * Gz, Bz = Hx * Gy - Hy * Gx;
      const A2 = Ax * Ax + Ay * Ay + Az * Az, B2 = Bx * Bx + By * By + Bz * Bz;
      const Gn = Math.sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
      const cosp = Ax * Bx + Ay * By + Az * Bz, sinp = -Gn * (Fx * Bx + Fy * By + Fz * Bz);
      const phi = Math.atan2(sinp, cosp);
      const n = this.tn[m], ang = n * phi - this.tph[m];
      et += this.tk[m] * (1 + Math.cos(ang));
      const dEdphi = -this.tk[m] * n * Math.sin(ang);
      const FG = Fx * Gx + Fy * Gy + Fz * Gz, HG = Hx * Gx + Hy * Gy + Hz * Gz;
      const a = Gn / A2, b = Gn / B2, c = FG / (A2 * Gn), d = HG / (B2 * Gn);
      // dphi/dr: i = -a A ; l = b B ; j = (a + c) A - d B ; k = -b B - c A + d B
      const s = -dEdphi;
      f[i] += s * (-a * Ax); f[i + 1] += s * (-a * Ay); f[i + 2] += s * (-a * Az);
      f[l] += s * (b * Bx); f[l + 1] += s * (b * By); f[l + 2] += s * (b * Bz);
      f[j] += s * ((a + c) * Ax - d * Bx); f[j + 1] += s * ((a + c) * Ay - d * By); f[j + 2] += s * ((a + c) * Az - d * Bz);
      f[k] += s * (-b * Bx - c * Ax + d * Bx); f[k + 1] += s * (-b * By - c * Ay + d * By); f[k + 2] += s * (-b * Bz - c * Az + d * Bz);
    }
    // ---- LJ + Coulomb (non-excluded pairs); merged into the GB loop when GB is on
    const pi = this.pi, pj = this.pj;
    for (let m = 0, np = this.gb ? 0 : pi.length; m < np; m++) {
      const i = 3 * pi[m], j = 3 * pj[m];
      const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
      const r2 = dx * dx + dy * dy + dz * dz, ir2 = 1 / r2, ir = Math.sqrt(ir2);
      const s6 = Math.pow(this.sig2[m] * ir2, 3), s12 = s6 * s6;
      elj += this.eps4[m] * (s12 - s6);
      const ecp = this.kqq[m] * ir; ec += ecp;
      const g = (this.eps4[m] * (12 * s12 - 6 * s6) + ecp) * ir2;
      f[i] += g * dx; f[i + 1] += g * dy; f[i + 2] += g * dz;
      f[j] -= g * dx; f[j + 1] -= g * dy; f[j + 2] -= g * dz;
    }
    // ---- 1-4 exceptions
    for (let m = 0, ne = this.ei.length; m < ne; m++) {
      const i = 3 * this.ei[m], j = 3 * this.ej[m];
      const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
      const r2 = dx * dx + dy * dy + dz * dz, ir2 = 1 / r2, ir = Math.sqrt(ir2);
      const s6 = Math.pow(this.esig2[m] * ir2, 3), s12 = s6 * s6;
      elj14 += this.eeps4[m] * (s12 - s6);
      const ecp = this.ekqq[m] * ir; ec14 += ecp;
      const g = (this.eeps4[m] * (12 * s12 - 6 * s6) + ecp) * ir2;
      f[i] += g * dx; f[i + 1] += g * dy; f[i + 2] += g * dz;
      f[j] -= g * dx; f[j + 1] -= g * dy; f[j + 2] -= g * dz;
    }
    // ---- GB-OBC2
    if (this.gb) {
      const or = this.or, sr = this.sr, rad = this.rad, q = this.q, I = this.I, R = this.R;
      const dRdpsi = this.dRdpsi, dEdR = this.dEdR, h = [0, 0];
      const PA = this.pa, PB = this.pb, NP = PA.length, pr = this.pr, pdIab = this.pdIab, pdIba = this.pdIba;
      const peps4 = this.peps4, psig2 = this.psig2, pkqq = this.pkqq;
      I.fill(0); dEdR.fill(0);
      // pass 1: Born integrals (and their r-derivatives, cached for pass 3)
      for (let p = 0; p < NP; p++) {
        const a = PA[p], b = PB[p], i = 3 * a, j = 3 * b;
        const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        pr[p] = r;
        hct(r, or[a], sr[b], h); I[a] += h[0]; pdIab[p] = h[1];
        hct(r, or[b], sr[a], h); I[b] += h[0]; pdIba[p] = h[1];
      }
      for (let a = 0; a < N; a++) {
        const psi = or[a] * I[a];
        const arg = ALPHA * psi - BETA * psi * psi + GAMMA * psi * psi * psi;
        const th = Math.tanh(arg);
        R[a] = 1 / (1 / or[a] - th / rad[a]);
        dRdpsi[a] = R[a] * R[a] * (1 - th * th) * (ALPHA - 2 * BETA * psi + 3 * GAMMA * psi * psi) / rad[a];
        const qq = this.gbPre * q[a] * q[a];
        egb += 0.5 * qq / R[a];
        dEdR[a] += -0.5 * qq / (R[a] * R[a]);
        const sa = this.ace[a] * Math.pow(rad[a] / R[a], 6);
        esa += sa;
        dEdR[a] += -6 * sa / R[a];
      }
      // pass 2: GB pair energy + LJ/Coulomb, explicit r-forces, and dE/dR
      const gbPre = this.gbPre;
      for (let p = 0; p < NP; p++) {
        const a = PA[p], b = PB[p], i = 3 * a, j = 3 * b;
        const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        const RR = R[a] * R[b], D = r2 / (4 * RR), ex = Math.exp(-D);
        const f2 = r2 + RR * ex, fgb = Math.sqrt(f2), f3 = f2 * fgb;
        const qq = gbPre * q[a] * q[b];
        egb += qq / fgb;
        // dP/dr = -qq r (1 - ex/4) / f^3  ->  force on a = -dP/dr * d/r
        let g = qq * (1 - 0.25 * ex) / f3;
        if (peps4[p] !== 0 || pkqq[p] !== 0) {
          const ir2 = 1 / r2, s6 = Math.pow(psig2[p] * ir2, 3), s12 = s6 * s6, ecp = pkqq[p] * Math.sqrt(ir2);
          elj += peps4[p] * (s12 - s6); ec += ecp;
          g += (peps4[p] * (12 * s12 - 6 * s6) + ecp) * ir2;
        }
        // chain rule through Born radii needs dE/dR first, so add it in pass 3
        f[i] += g * dx; f[i + 1] += g * dy; f[i + 2] += g * dz;
        f[j] -= g * dx; f[j + 1] -= g * dy; f[j + 2] -= g * dz;
        const c = -0.5 * qq * ex * (1 + D) / f3;   // dP/dRa = c * Rb
        dEdR[a] += c * R[b]; dEdR[b] += c * R[a];
      }
      // pass 3: chain rule through the Born radii (cached r and dI/dr)
      for (let a = 0; a < N; a++) dRdpsi[a] *= dEdR[a] * or[a];
      for (let p = 0; p < NP; p++) {
        const a = PA[p], b = PB[p];
        const dEdr = dRdpsi[a] * pdIab[p] + dRdpsi[b] * pdIba[p];
        if (dEdr === 0) continue;
        const i = 3 * a, j = 3 * b, g = -dEdr / pr[p];
        const dx = x[i] - x[j], dy = x[i + 1] - x[j + 1], dz = x[i + 2] - x[j + 2];
        f[i] += g * dx; f[i + 1] += g * dy; f[i + 2] += g * dz;
        f[j] -= g * dx; f[j + 1] -= g * dy; f[j + 2] -= g * dz;
      }
    }
    const t = this.terms;
    t.bond = eb; t.angle = ea; t.torsion = et; t.lj = elj; t.coulomb = ec; t.lj14 = elj14; t.coulomb14 = ec14;
    t.gb_polar = egb; t.gb_nonpolar = esa;
    this.energy = eb + ea + et + elj + ec + elj14 + ec14 + egb + esa;
    return this.energy;
  };

  // ---------------------------------------------------------------------------
  // Langevin BAOAB dynamics
  // ---------------------------------------------------------------------------
  function gaussian(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function Dynamics(engine, x0, opts) {
    opts = opts || {};
    this.e = engine;
    const N = engine.N;
    this.x = Float64Array.from(x0.flat ? x0.flat() : x0);
    this.v = new Float64Array(3 * N);
    this.dt = opts.dt || 0.001;
    this.T = opts.temperature || 300;
    this.gamma = opts.friction || 1.0;
    this.rng = mulberry32(opts.seed || 1);
    this.time = 0; this.steps = 0;
    this.thermalize(this.T);
    engine.compute(this.x);
  }

  Dynamics.prototype.thermalize = function (T) {
    const N = this.e.N, m = this.e.mass;
    for (let a = 0; a < N; a++) {
      const s = Math.sqrt(KB * T / m[a]);
      for (let k = 0; k < 3; k++) this.v[3 * a + k] = s * gaussian(this.rng);
    }
  };

  Dynamics.prototype.step = function (n) {
    const N = this.e.N, m = this.e.mass, x = this.x, v = this.v, dt = this.dt;
    const c1 = Math.exp(-this.gamma * dt), c2 = Math.sqrt(1 - c1 * c1);
    let f = this.e.f;
    for (let s = 0; s < n; s++) {
      for (let a = 0; a < N; a++) {
        const im = 1 / m[a], sv = c2 * Math.sqrt(KB * this.T * im);
        for (let k = 0; k < 3; k++) {
          const p = 3 * a + k;
          v[p] += 0.5 * dt * f[p] * im;                // B
          x[p] += 0.5 * dt * v[p];                     // A
          v[p] = c1 * v[p] + sv * gaussian(this.rng);  // O
          x[p] += 0.5 * dt * v[p];                     // A
        }
      }
      this.e.compute(x);
      f = this.e.f;
      for (let a = 0; a < N; a++) {
        const im = 1 / m[a];
        for (let k = 0; k < 3; k++) { const p = 3 * a + k; v[p] += 0.5 * dt * f[p] * im; }  // B
      }
      this.time += dt; this.steps++;
    }
  };

  Dynamics.prototype.kineticEnergy = function () {
    let ek = 0; const m = this.e.mass, v = this.v;
    for (let a = 0; a < this.e.N; a++) ek += 0.5 * m[a] * (v[3 * a] ** 2 + v[3 * a + 1] ** 2 + v[3 * a + 2] ** 2);
    return ek;
  };

  Dynamics.prototype.temperature = function () {
    return 2 * this.kineticEnergy() / ((3 * this.e.N - 3) * KB);
  };

  const api = { Engine, Dynamics, KB, ONE_4PI_EPS0 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.VHMD = api;
})(typeof window !== "undefined" ? window : globalThis);
