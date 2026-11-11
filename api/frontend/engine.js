/**
 * Meridian wire fraud engine (browser / Node).
 *
 * Scores a wire 0-1 with the weights trained by `python -m ml.train`
 * (loaded from window.MODEL_WEIGHTS in model_weights.js, or model_weights.json in Node),
 * explains the top 3 reasons, and routes it to release / confirm / hold.
 *
 * Gives exactly the same results as ml/scoring.py in the API.
 *
 *   const r = WireRisk.score({ amount: 85000, usualAmount: 9000, remoteAccess: true,
 *                              beneficiaryAgeDays: 12, firstTimePayee: true, channel: 'online' });
 *   r.score, r.lane, r.reasons, r.steps
 *
 * Policy: channel 'voice' is ALWAYS routed to confirm, whatever the score.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../artifacts/model_weights.json'));
  else root.WireRisk = factory(root.MODEL_WEIGHTS);
})(typeof self !== 'undefined' ? self : this, function (W) {
  'use strict';
  if (!W) throw new Error('Model weights not loaded. Run `python -m ml.train` to create frontend/model_weights.js.');

  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const num = (v, d) => { const n = parseFloat(v); return Number.isFinite(n) ? n : d; };
  const plural = n => (n === 1 ? '' : 's');

  // How to read each feature from a wire, keyed by jsKey (order comes from the weights file).
  const GET = {
    amountRatio: t => Math.max(t.amountVsUsual != null ? num(t.amountVsUsual, 1) : Math.max(num(t.amount, 1), 1) / Math.max(num(t.usualAmount, 1), 1), 0.01),
    debtToIncome: t => clamp(num(t.debtToIncome, 0), 0, 1),
    typingMismatch: t => clamp(num(t.typingMismatch, 0), 0, 1),
    beneficiaryAgeDays: t => Math.max(1, Math.round(num(t.beneficiaryAgeDays, 1))),
    beneficiarySenders30d: t => Math.max(1, Math.round(num(t.beneficiarySenders30d, 1))),
  };
  const TEXT = {
    amountRatio: x => `Amount is ${x.toFixed(1)}× this customer’s usual wire`,
    outsideHours: () => 'Sent outside this customer’s usual hours',
    debtToIncome: x => `Debt-to-income is ${Math.round(x * 100)}%`,
    newDevice: () => 'New device or IP address for this customer',
    typingMismatch: () => 'Typing rhythm doesn’t match this customer',
    remoteAccess: () => 'Remote-access software running in session',
    simSwap7d: () => 'SIM swap on the customer’s phone in the last 7 days',
    muleListHit: () => 'Beneficiary is on the consortium mule list',
    beneficiaryAgeDays: x => `Beneficiary account opened ${x} day${plural(x)} ago`,
    bankDetailsChanged30d: () => 'Payee’s bank details changed in the last 30 days',
    highRiskDestination: () => 'High-risk or new destination country',
    firstTimePayee: () => 'First wire to this payee',
    beneficiarySenders30d: x => `${x} different senders paid this beneficiary in 30 days`,
  };
  const TF = { id: x => x, log: x => Math.log(x), neglog1p: x => -Math.log(x + 1) };

  function isVoice(tx) {
    return !!tx && (String(tx.channel || '').toLowerCase() === 'voice' || tx.voiceRequest === true);
  }
  function decide(s, th) {
    th = Object.assign({}, W.thresholds, th);
    return s < th.release ? 'release' : s < th.hold ? 'confirm' : 'hold';
  }
  function nextSteps(lane, tx) {
    tx = tx || {};
    if (lane === 'confirm' && isVoice(tx)) return [
      { title: 'Passkey on the customer’s verified device', detail: 'Voice requests are always confirmed out of band; the voice is never proof of identity.' },
      { title: 'Customer confirms the payee name in the app', detail: 'Typed on the verified device, not spoken on the call, so a cloned voice can’t pass it.' },
      { title: 'Release, or move to hold if either step fails', detail: 'Outcome is logged as a label.' }];
    if (lane === 'release') return [{ title: 'Release automatically', detail: 'No extra step for the customer. Score and reasons are logged.' }];
    if (lane === 'confirm') return [
      { title: tx.newDevice ? 'Passkey on the customer’s verified device' : 'Passkey on the current device', detail: 'Approval is pushed to the registered device when the session is on a new device or IP.' },
      { title: 'Customer confirms the payee name', detail: 'Matched to the beneficiary account name before release.' },
      { title: 'Release, or move to hold if either step fails', detail: 'Outcome is logged as a label.' }];
    return [
      { title: 'Hold the wire and step up identity (Okta)', detail: 'Phishing-resistant re-verification.' },
      { title: 'Call the trusted contact on file', detail: 'Only a contact unchanged for at least ~2 weeks is used.' },
      { title: 'Analyst decides; otherwise blocked', detail: 'If identity can’t be confirmed, the customer sends in person.' }];
  }

  function score(tx, opts) {
    tx = tx || {}; opts = opts || {};
    const raw = W.features.map(f => (GET[f.jsKey] ? GET[f.jsKey](tx) : (tx[f.jsKey] ? 1 : 0)));
    const contrib = raw.map((x, j) => W.w[j] * (TF[W.features[j].transform](x) - W.mu[j]) / W.sd[j]);
    const modelScore = 1 / (1 + Math.exp(-(W.b + W.shift + contrib.reduce((a, c) => a + c, 0))));
    const overridden = typeof opts.override === 'number' && Number.isFinite(opts.override);
    const s = overridden ? clamp(opts.override, 0, 1) : modelScore;
    const scoreLane = decide(s, opts.thresholds);
    const voice = isVoice(tx);
    const lane = voice ? 'confirm' : scoreLane;
    const reasons = contrib
      .map((c, j) => ({ key: W.features[j].jsKey, label: W.features[j].label, text: TEXT[W.features[j].jsKey](raw[j]), contribution: Math.round(c * 1000) / 1000 }))
      .filter(r => r.contribution > 0).sort((a, b) => b.contribution - a.contribution).slice(0, 3);
    return {
      score: Math.round(s * 1e4) / 1e4, modelScore: Math.round(modelScore * 1e4) / 1e4, overridden,
      lane, scoreLane, policy: voice ? 'voice_always_confirm' : null, channel: voice ? 'voice' : (tx.channel || 'online'),
      reasons, steps: nextSteps(lane, tx), modelVersion: W.version, scoredAt: new Date().toISOString(),
    };
  }

  return { VERSION: W.version, DEFAULT_THRESHOLDS: W.thresholds, METRICS: W.metrics, FEATURES: W.features, score, decide, nextSteps, isVoice };
});
