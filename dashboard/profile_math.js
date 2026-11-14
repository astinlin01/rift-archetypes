/*
 * Browser copy of the profile math, so the static dashboard can recompute everything when
 * the game-window / champion filters change. Each function mirrors a tested Python
 * function (named in its comment); tests/test_dashboard_math.py runs both on the same
 * inputs through Node and requires matching answers.
 *
 * Plain ES5-style JavaScript with no dependencies: build_dashboard.py inlines this file
 * into the page, and Node loads it via module.exports for the tests.
 */
(function (root) {
  'use strict';

  function isNum(v) {
    return v !== null && v !== undefined && typeof v === 'number' && isFinite(v);
  }

  function mean(values) {
    var nums = values.filter(isNum);
    if (!nums.length) return null;
    return nums.reduce(function (a, b) { return a + b; }, 0) / nums.length;
  }

  function sampleStd(values) {
    var nums = values.filter(isNum);
    if (nums.length < 2) return null;
    var m = mean(nums);
    var ss = nums.reduce(function (a, v) { return a + (v - m) * (v - m); }, 0);
    return Math.sqrt(ss / (nums.length - 1));
  }

  // Percentile of `value` among reference player averages, given only evenly spaced
  // percentile cutoffs (breakpoints[i] = the (100*i/(n-1))th percentile). Linear
  // between cutoffs. A value equal to a run of identical cutoffs (common for stats that
  // are often exactly 0) lands in the middle of the run -- ties count half, like
  // scipy's percentileofscore(kind="mean") on the Python side.
  function percentileFromBreakpoints(value, breakpoints) {
    var n = breakpoints.length;
    var step = 100 / (n - 1);
    if (value < breakpoints[0]) return 0;
    if (value > breakpoints[n - 1]) return 100;
    var first = -1, last = -1;
    for (var i = 0; i < n; i++) {
      if (breakpoints[i] === value) {
        if (first < 0) first = i;
        last = i;
      }
    }
    if (first >= 0) return ((first + last) / 2) * step;
    for (var j = 0; j < n - 1; j++) {
      if (value < breakpoints[j + 1]) {
        var lo = breakpoints[j], hi = breakpoints[j + 1];
        return (j + (value - lo) / (hi - lo)) * step;
      }
    }
    return 100;
  }

  // style_profile.player_profiles, for one player: mean of each trait over the games
  // that have it, plus how many games that was.
  function traitProfile(games, traits) {
    var profile = {}, counts = {};
    traits.forEach(function (trait) {
      var values = games.map(function (g) { return g[trait]; }).filter(isNum);
      counts[trait] = values.length;
      profile[trait] = values.length ? mean(values) : null;
    });
    return { profile: profile, counts: counts };
  }

  // playstyle_space._signature_match: cosine similarity between the profile and an
  // archetype signature, using the norm of the *whole* profile.
  function signatureMatch(profile, signature) {
    var present = Object.keys(signature).filter(function (t) { return isNum(profile[t]); });
    if (!present.length) return -1;
    var dot = 0, sigNorm = 0;
    present.forEach(function (t) {
      dot += signature[t] * profile[t];
      sigNorm += signature[t] * signature[t];
    });
    var fullNorm = 0;
    Object.keys(profile).forEach(function (t) {
      if (isNum(profile[t])) fullNorm += profile[t] * profile[t];
    });
    if (fullNorm === 0) return -1;
    return dot / (Math.sqrt(sigNorm) * Math.sqrt(fullNorm));
  }

  // style_profile.soft_labels: top-N [archetype, match %], best first, clamped to 0-100.
  function softLabels(profile, archetypes, top) {
    var matches = Object.keys(archetypes).map(function (name) {
      var pct = signatureMatch(profile, archetypes[name]) * 100;
      return [name, Math.min(100, Math.max(0, pct))];
    });
    matches.sort(function (a, b) { return b[1] - a[1]; });
    return matches.slice(0, top === undefined ? 2 : top);
  }

  // style_profile.profile_reliability (Spearman-Brown).
  function profileReliability(singleGame, games) {
    if (games <= 0 || !isNum(singleGame)) return 0;
    return games * singleGame / (1 + (games - 1) * singleGame);
  }

  // style_profile.confidence_label.
  function confidenceLabel(reliability) {
    if (reliability >= 0.7) return 'high';
    if (reliability >= 0.5) return 'medium';
    return 'low';
  }

  // performance.own_sample_status.
  function ownSampleStatus(games) {
    if (games < 5) return 'insufficient';
    return games < 10 ? 'low' : 'ok';
  }

  // champion_pool.effective_champions: 1 / unbiased HHI, capped at champions played.
  function effectiveChampions(counts) {
    var values = Object.keys(counts).map(function (k) { return counts[k]; }).filter(function (v) { return v > 0; });
    var total = values.reduce(function (a, b) { return a + b; }, 0);
    if (total < 2) return null;
    var samePairs = values.reduce(function (a, n) { return a + n * (n - 1); }, 0);
    if (samePairs === 0) return values.length;
    return Math.min(total * (total - 1) / samePairs, values.length);
  }

  // champion_pool.top_champion: [name, share]; first-listed wins ties.
  function topChampion(counts) {
    var best = null, total = 0;
    Object.keys(counts).forEach(function (name) {
      total += counts[name];
      if (best === null || counts[name] > counts[best]) best = name;
    });
    return best === null ? null : [best, counts[best] / total];
  }

  // champion_pool.class_mix, from each game's already-resolved class label.
  function classMix(labels) {
    var mix = {};
    labels.forEach(function (label) { mix[label] = (mix[label] || 0) + 1; });
    Object.keys(mix).forEach(function (label) { mix[label] /= labels.length; });
    return mix;
  }

  // performance.biggest_gaps. curves[stat] = {breakpoints, median} of the reference
  // players' averages; gapFeatures[stat] = [theme, direction].
  var GAP_INTERVAL_Z = 1.2816;  // 80% two-sided
  function biggestGaps(games, curves, gapFeatures, top) {
    var bestPerTheme = {}, themeOrder = [];
    Object.keys(gapFeatures).forEach(function (feature) {
      var theme = gapFeatures[feature][0], direction = gapFeatures[feature][1];
      var curve = curves[feature];
      if (!curve) return;
      var values = games.map(function (g) { return g[feature]; }).filter(isNum);
      if (!values.length) return;
      var avg = mean(values);
      var raw = percentileFromBreakpoints(avg, curve.breakpoints);
      var percentile = direction > 0 ? raw : 100 - raw;
      if (percentile >= 50) return;
      var sd = sampleStd(values);
      var se = sd === null ? Infinity : sd / Math.sqrt(values.length);
      if (Math.abs(avg - curve.median) <= GAP_INTERVAL_Z * se) return;  // within noise
      var gap = {
        feature: feature, theme: theme, percentile: percentile,
        your_avg: avg, reference_median: curve.median, n_games: values.length
      };
      if (!(theme in bestPerTheme)) themeOrder.push(theme);
      if (!(theme in bestPerTheme) || percentile < bestPerTheme[theme].percentile) bestPerTheme[theme] = gap;
    });
    var gaps = themeOrder.map(function (t) { return bestPerTheme[t]; });
    gaps.sort(function (a, b) { return a.percentile - b.percentile; });
    return gaps.slice(0, top === undefined ? 3 : top);
  }

  // performance.gap_size: a gap in game terms. Per-minute stats -> per game of the role's
  // typical length; 0-1 shares -> percentage points; else the plain difference.
  function gapSize(feature, yourAvg, median, gameMinutes, percentFeatures) {
    var diff = Math.abs(median - yourAvg);
    if (/_per_min(ute)?$/.test(feature)) return { amount: diff * gameMinutes, per_game: true };
    if ((percentFeatures || []).indexOf(feature) !== -1) return { amount: diff * 100, per_game: false };
    return { amount: diff, per_game: false };
  }

  // Style map default axes: the two traits furthest from the 50th (your most distinctive),
  // most distinctive first. rows = [{trait, percentile}]. null if fewer than two.
  function defaultMapAxes(rows) {
    var usable = rows.filter(function (r) { return isNum(r.percentile); });
    if (usable.length < 2) return null;
    usable = usable.slice().sort(function (a, b) {
      return Math.abs(b.percentile - 50) - Math.abs(a.percentile - 50);
    });
    return [usable[0].trait, usable[1].trait];
  }

  var ProfileMath = {
    mean: mean,
    percentileFromBreakpoints: percentileFromBreakpoints,
    traitProfile: traitProfile,
    signatureMatch: signatureMatch,
    softLabels: softLabels,
    profileReliability: profileReliability,
    confidenceLabel: confidenceLabel,
    ownSampleStatus: ownSampleStatus,
    effectiveChampions: effectiveChampions,
    topChampion: topChampion,
    classMix: classMix,
    biggestGaps: biggestGaps,
    gapSize: gapSize,
    defaultMapAxes: defaultMapAxes
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = ProfileMath;
  else root.ProfileMath = ProfileMath;
})(this);
