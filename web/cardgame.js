'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  gamePhase: 'menu',   // 'menu' | 'play' | 'gameover' | 'series-over'
  showRules: false,    // rules overlay sits on top without changing gamePhase
  rulesPage: 1,
  player: [],
  cpu: [],
  current: [],
  currType: [0, 0, 0], // [type, length, rank] of the trick on the table
  turn: true,          // true = player's turn
  playerScore: 0,
  cpuScore: 0,
  lastWinner: null,    // 'player' | 'cpu' | null (for series initiative)
  gameResult: null,
  invalidMove: false,
  /** Rank-count array of every card played so far (engine rank order). */
  discard: new Array(13).fill(0),
  /** @type {{ who: 'player'|'cpu'|'system', text: string }[]} */
  history: [],
};

// ─── Engine rank mapping ────────────────────────────────────────────────────
// Engine rank order: index 0='3' … 7='10', 8='J', 9='Q', 10='K', 11='A', 12='2'.
function numToIdx(n) {
  if (n === 1) return 11;  // Ace
  if (n === 2) return 12;  // deuce
  return n - 3;            // 3..13 -> 0..10
}
function idxToNum(i) {
  if (i === 11) return 1;
  if (i === 12) return 2;
  return i + 3;
}
function countsFromCards(cards) {
  const c = new Array(13).fill(0);
  for (const card of cards) c[numToIdx(card.number)]++;
  return c;
}

// ─── Card Utilities ───────────────────────────────────────────────────────────
function rankLabel(n) {
  if (n === 1)  return 'A';
  if (n === 11) return 'J';
  if (n === 12) return 'Q';
  if (n === 13) return 'K';
  return String(n);
}

// suit string '0'=♠ '1'=♥ '2'=♣ '3'=♦
function suitSymbol(s) {
  return ['♠', '♥', '♣', '♦'][Number(s)];
}

function isRed(suit) {
  return suit === '1' || suit === '3';
}

// ─── Game Logic ───────────────────────────────────────────────────────────────

// Returns true if rank a strictly beats rank b in Big 2 order (3 < 4 < … < K < A < 2)
function trumps(a, b) {
  if (a === b) return false;
  if (a === 2) return true;
  if (a === 1) return b !== 2 && b !== 1;
  if (b === 2 || b === 1) return false;
  return a > b;
}

// Position in straight order: 3→1, 4→2, …, K→11, A→12, 2→13
function straightPos(n) {
  if (n >= 3) return n - 2;
  if (n === 1) return 12;
  return 13; // 2
}

// Sort hand ascending in Big 2 order (3 first, 2 last); returns a new array
function sortHand(lst) {
  const sorted = [...lst];
  for (let i = sorted.length - 1; i > 0; i--) {
    for (let j = 0; j < i; j++) {
      if (trumps(sorted[j].number, sorted[j + 1].number)) {
        [sorted[j], sorted[j + 1]] = [sorted[j + 1], sorted[j]];
      }
    }
  }
  return sorted;
}

function allSame(lst) {
  return lst.every(c => c.number === lst[0].number);
}

function isAceBomb(lst) {
  return lst.length === 3 && lst.every(c => c.number === 1);
}

// Returns top-card number if sorted cards form a valid straight (length ≥ 5), else null.
// Handles normal consecutive order and 2 used as a low card (2,3,4,5,6).
// A,2,3,4,5 is INVALID per the rules.
function checkStraight(sorted) {
  const n = sorted.length;
  if (n < 5) return null;

  // Case 1: normal consecutive in straight-position order
  let ok = true;
  for (let i = 1; i < n; i++) {
    if (straightPos(sorted[i].number) !== straightPos(sorted[i - 1].number) + 1) {
      ok = false;
      break;
    }
  }
  if (ok) return sorted[n - 1].number;

  // Case 2: 2 used as low card → sorted = [3, 4, …, k, 2]
  // A,2,3,4,5 is excluded here because A cannot precede 2 in a low-2 straight.
  if (sorted[n - 1].number === 2) {
    let lowOk = true;
    for (let i = 0; i < n - 1; i++) {
      if (sorted[i].number !== i + 3) { lowOk = false; break; }
    }
    if (lowOk) return sorted[n - 2].number; // rank = highest non-2 card
  }
  return null;
}

// Returns rank of top pair if sorted cards form `pairs` consecutive pairs, else null.
function checkDoubleStraight(sorted, pairs) {
  const n = sorted.length;
  if (n !== pairs * 2) return null;
  for (let i = 0; i < n; i += 2) {
    if (sorted[i].number !== sorted[i + 1].number) return null;
  }
  for (let i = 2; i < n; i += 2) {
    if (straightPos(sorted[i].number) !== straightPos(sorted[i - 2].number) + 1) return null;
  }
  return sorted[n - 2].number;
}

// Returns rank of top triple if sorted cards form `triples` consecutive triples, else null.
// Triple Ace = bomb, so it may not appear in a triple straight.
function checkTripleStraight(sorted, triples) {
  const n = sorted.length;
  if (n !== triples * 3) return null;
  for (let i = 0; i < n; i += 3) {
    if (sorted[i].number !== sorted[i + 1].number || sorted[i + 1].number !== sorted[i + 2].number) return null;
    if (sorted[i].number === 1) return null; // triple ace is a bomb
  }
  for (let i = 3; i < n; i += 3) {
    if (straightPos(sorted[i].number) !== straightPos(sorted[i - 3].number) + 1) return null;
  }
  return sorted[n - 1].number;
}

// Returns the rank of the triple if sorted 5-card hand is a full house, else null.
function checkFullHouse(sorted) {
  if (sorted.length !== 5) return null;
  // aaabb – triple at 0-2, pair at 3-4
  if (sorted[0].number === sorted[1].number && sorted[1].number === sorted[2].number &&
      sorted[3].number === sorted[4].number && sorted[0].number !== sorted[3].number &&
      sorted[0].number !== 1) {
    return sorted[0].number;
  }
  // aabbb – pair at 0-1, triple at 2-4
  if (sorted[0].number === sorted[1].number &&
      sorted[2].number === sorted[3].number && sorted[3].number === sorted[4].number &&
      sorted[0].number !== sorted[2].number && sorted[2].number !== 1) {
    return sorted[2].number;
  }
  return null;
}

// Returns [type, length, rank]:
//   0 = fresh/empty  1 = single/straight  2 = pair/sisters
//   3 = triple/triple-straight  4 = bomb  5 = full house
//   [-1,-1,-1] = invalid selection
function playType(lst) {
  const sorted = sortHand(lst);
  const n = sorted.length;

  if (n === 0) return [0, 0, 0];
  if (n === 1) return [1, 1, sorted[0].number];

  if (n === 2) {
    if (sorted[0].number === sorted[1].number) return [2, 1, sorted[0].number];
    return [-1, -1, -1];
  }

  if (n === 3) {
    if (allSame(sorted)) {
      return sorted[0].number === 1 ? [4, 0, 1] : [3, 1, sorted[0].number];
    }
    return [-1, -1, -1];
  }

  if (n === 4) {
    if (allSame(sorted)) return [4, 0, sorted[0].number]; // 4-of-a-kind bomb
    // A-bomb with one add-on: [x,A,A,A] or [A,A,A,2]
    if (isAceBomb(sorted.slice(0, 3)) || isAceBomb(sorted.slice(1, 4))) return [4, 1, 1];
    const ds = checkDoubleStraight(sorted, 2);
    if (ds !== null) return [2, 2, ds];
    return [-1, -1, -1];
  }

  if (n === 5) {
    // 4-of-a-kind bomb with one add-on
    if (allSame(sorted.slice(0, 4))) return [4, 1, sorted[0].number];
    if (allSame(sorted.slice(1, 5))) return [4, 1, sorted[1].number];
    const s = checkStraight(sorted);
    if (s !== null) return [1, 5, s];
    const fh = checkFullHouse(sorted);
    if (fh !== null) return [5, 0, fh];
    return [-1, -1, -1];
  }

  // n >= 6: check straight, then sisters, then triple straight
  const s = checkStraight(sorted);
  if (s !== null) return [1, n, s];
  if (n % 2 === 0) {
    const ds = checkDoubleStraight(sorted, n / 2);
    if (ds !== null) return [2, n / 2, ds];
  }
  if (n % 3 === 0) {
    const ts = checkTripleStraight(sorted, n / 3);
    if (ts !== null) return [3, n / 3, ts];
  }
  return [-1, -1, -1];
}

// Returns true if the play described by `type` is legal given the current trick.
function validPlay(type) {
  if (type[0] === -1) return false;
  const ct = state.currType;
  if (ct[0] === 0) return true;                                                        // fresh trick
  if (ct[0] === type[0] && ct[1] === type[1] && trumps(type[2], ct[2])) return true;  // same type+len, higher rank
  if (type[0] === 4 && !(ct[0] === 4 && !trumps(type[2], ct[2]))) return true;        // bomb beats non-bomb or lower bomb
  return false;
}

function getPlay() {
  return playType(state.player.filter(c => c.selected));
}

// ─── Deck / Deal ─────────────────────────────────────────────────────────────
function buildAndShuffle() {
  const deck = [];
  // 3–K: 4 of each suit
  for (let n = 3; n <= 13; n++) {
    for (let s = 0; s < 4; s++) {
      deck.push({ number: n, suit: String(s), selected: false });
    }
  }
  // 3 aces (remove one ace as per the rules)
  for (let s = 0; s < 3; s++) {
    deck.push({ number: 1, suit: String(s), selected: false });
  }
  // 1 two (remove three 2s as per the rules)
  deck.push({ number: 2, suit: '0', selected: false });

  // Fisher-Yates shuffle
  for (let i = deck.length - 1; i > 0; i--) {
    const r = Math.floor(Math.random() * (i + 1));
    [deck[i], deck[r]] = [deck[r], deck[i]];
  }
  return deck;
}

function findCard(n, s, isPlayer) {
  const hand = isPlayer ? state.player : state.cpu;
  return hand.some(c => c.number === n && c.suit === s);
}

function deal() {
  const deck = buildAndShuffle();
  state.player = sortHand(deck.slice(0, 16));
  state.cpu    = sortHand(deck.slice(16, 32));
  // third pile (32-47) is unused

  // Initiative: previous series-game winner leads; first game uses lowest spade rule
  if (state.lastWinner === 'player') { state.turn = true;  return; }
  if (state.lastWinner === 'cpu')    { state.turn = false; return; }

  // First game: find lowest spade (3♠ → 4♠ → … → 2♠)
  for (let i = 3; i <= 15; i++) {
    const n = ((i - 1) % 13) + 1;
    if (findCard(n, '0', true))  { state.turn = true;  return; }
    if (findCard(n, '0', false)) { state.turn = false; return; }
  }
  // Fallback: lowest heart (very unlikely — all spades in unused pile)
  for (let i = 3; i <= 6; i++) {
    if (findCard(i, '1', true))  { state.turn = true;  return; }
    if (findCard(i, '1', false)) { state.turn = false; return; }
  }
}

// ─── Play Actions ─────────────────────────────────────────────────────────────
function bonusPoints(remaining) {
  if (remaining >= 16) return 50;
  if (remaining === 15) return 40;
  if (remaining === 14) return 30;
  if (remaining === 13) return 20;
  return 0;
}

function endGame(winner) {
  const loserRemaining = winner === 'player' ? state.cpu.length : state.player.length;
  const bonus  = bonusPoints(loserRemaining);
  // A high card-count bonus *replaces* the base score — it does not add to it.
  const points = bonus > 0 ? bonus : loserRemaining;

  pushHistory('system', winner === 'player' ? 'You won the hand.' : 'CPU won the hand.');

  state.lastWinner = winner;
  if (winner === 'player') state.playerScore += points;
  else                     state.cpuScore    += points;

  const seriesWon    = state.playerScore >= 50 || state.cpuScore >= 50;
  const seriesWinner = state.playerScore >= 50 ? 'player'
                     : state.cpuScore    >= 50 ? 'cpu' : null;

  state.gameResult = { winner, loserRemaining, bonus, points, seriesWon, seriesWinner };
  // Pause on the final board (last move + revealed hands visible) until the
  // player clicks through to the result popup.
  state.gamePhase  = 'review';
}

function pushHistory(who, text) {
  state.history.push({ who, text });
  if (state.history.length > 200) state.history.shift();
}

/** Compact display of played cards (sorted by rank). */
function cardsShortString(cards) {
  if (!cards.length) return '';
  const sorted = sortHand(cards);
  let s = sorted.map(c => rankLabel(c.number) + suitSymbol(c.suit)).join(' ');
  if (s.length > 96) s = s.slice(0, 93) + '…';
  return s;
}

// Commits the currently-selected cards for the given player, updates state.
function playCurrent(isPlayer) {
  const hand      = isPlayer ? state.player : state.cpu;
  const played    = hand.filter(c => c.selected);
  const remaining = hand.filter(c => !c.selected);

  played.forEach(c => { c.selected = false; state.discard[numToIdx(c.number)]++; });
  state.current  = played;
  state.currType = playType(played);

  const label = isPlayer ? 'You' : 'CPU';
  pushHistory(isPlayer ? 'player' : 'cpu',
    `${label} played ${typeName(state.currType)} — ${cardsShortString(played)}`);

  if (isPlayer) {
    state.player = remaining;
    if (state.player.length === 0) { endGame('player'); return; }
  } else {
    state.cpu = remaining;
    if (state.cpu.length === 0) { endGame('cpu'); return; }
  }
  state.turn = !isPlayer;
}

// ─── CPU AI (typed-search WASM engine) ──────────────────────────────────────

// Selects `cards` (a 13-element rank-count vector from the engine) out of the
// CPU's hand by marking matching cards (suit-agnostic). Returns true if the
// full requested combination was found.
function selectCpuCards(cards) {
  let ok = true;
  for (let idx = 0; idx < 13; idx++) {
    let need = cards[idx];
    if (need === 0) continue;
    const num = idxToNum(idx);
    for (const card of state.cpu) {
      if (need === 0) break;
      if (card.number === num && !card.selected) { card.selected = true; need--; }
    }
    if (need > 0) ok = false;  // engine asked for cards we don't hold (shouldn't happen)
  }
  return ok;
}

// Asks the WASM engine for the CPU's move and applies it (play or pass).
function cpuChoose() {
  const hand    = countsFromCards(state.cpu);
  const discard = state.discard.slice();
  const oppCount = state.player.length;
  const fresh   = state.currType[0] === 0;
  const trick   = fresh ? new Array(13).fill(0) : countsFromCards(state.current);

  const { moveId, cards } = Big2.selectMove(hand, discard, oppCount, trick);
  const total = cards.reduce((a, b) => a + b, 0);

  if (moveId === 0 || total === 0 || !selectCpuCards(cards)) {
    // Pass (only legal when responding to a trick; on a fresh trick the engine
    // always returns a real move, so this branch means "pass").
    state.cpu.forEach(c => { c.selected = false; });
    pushHistory('cpu', 'CPU passed');
    state.current  = [];
    state.currType = [0, 0, 0];
    state.turn     = true;
    return;
  }
  playCurrent(false);
}

// ─── Rendering ────────────────────────────────────────────────────────────────

function makeCardEl(card, small, faceDown) {
  const el = document.createElement('div');
  el.className = 'card' + (small ? ' card-small' : '');

  if (faceDown) {
    el.classList.add('card-back');
    return el;
  }

  if (isRed(card.suit)) el.classList.add('red');
  if (card.selected)    el.classList.add('selected');

  const r = rankLabel(card.number);
  const s = suitSymbol(card.suit);
  el.innerHTML =
    `<span class="corner tl"><span class="r">${r}</span><span class="s">${s}</span></span>` +
    `<span class="center-suit">${s}</span>` +
    `<span class="corner br"><span class="r">${r}</span><span class="s">${s}</span></span>`;
  return el;
}

function renderHand(cards, container, faceDown, onCardClick) {
  container.innerHTML = '';
  if (!cards.length) return;
  const fan = document.createElement('div');
  fan.className = 'hand-fan';
  cards.forEach((card, i) => {
    const el = makeCardEl(card, false, faceDown);
    el.dataset.index = String(i);
    if (onCardClick) el.addEventListener('click', () => onCardClick(i));
    fan.appendChild(el);
  });
  container.appendChild(fan);
}

function renderTrick(cards, container) {
  container.innerHTML = '';
  if (!cards.length) return;
  const row = document.createElement('div');
  row.className = 'trick-cards';
  cards.forEach(card => row.appendChild(makeCardEl(card, true, false)));
  container.appendChild(row);
}

function typeName(ct) {
  if (ct[0] === 1 && ct[1] === 1) return 'Single';
  if (ct[0] === 1) return `${ct[1]}-card Straight`;
  if (ct[0] === 2 && ct[1] === 1) return 'Pair';
  if (ct[0] === 2) return `${ct[1]}-pair Sisters`;
  if (ct[0] === 3 && ct[1] === 1) return 'Triple';
  if (ct[0] === 3) return `${ct[1]}-triple Straight`;
  if (ct[0] === 4) return 'Bomb';
  if (ct[0] === 5) return 'Full House';
  return '';
}

function renderBoard() {
  if (state.gamePhase !== 'play' && state.gamePhase !== 'review') return;
  const reviewing = state.gamePhase === 'review';

  // Scores
  document.getElementById('player-score').textContent = state.playerScore;
  document.getElementById('cpu-score').textContent    = state.cpuScore;

  // CPU hand (face-down during play; revealed face-up at hand's end)
  renderHand(state.cpu, document.getElementById('cpu-hand'), !reviewing, null);
  document.getElementById('cpu-count').textContent = state.cpu.length;

  // Trick area
  renderTrick(state.current, document.getElementById('current-trick'));
  const label = document.getElementById('trick-label');
  if (reviewing) {
    const winner = state.gameResult && state.gameResult.winner === 'player' ? 'You' : 'CPU';
    label.textContent = `Hand over — ${winner} played the final ${typeName(state.currType) || 'card'}`;
  } else if (state.currType[0] === 0) {
    label.textContent = state.turn ? 'Your lead — play any combination' : 'CPU is leading…';
  } else {
    label.textContent = typeName(state.currType) + ' on the table';
  }

  // Does the player have a legal response, or are they forced to pass?
  const yourTurn = state.turn && state.gamePhase === 'play';
  let mustPass = false;
  if (yourTurn && state.currType[0] !== 0 && Big2.isReady()) {
    mustPass = Big2.legalMoveCount(countsFromCards(state.player),
                                   countsFromCards(state.current)) === 0;
  }

  // Turn indicator
  const ind = document.getElementById('turn-indicator');
  if (reviewing) {
    ind.textContent = 'Hand over';
    ind.className = 'review-turn';
  } else if (mustPass) {
    ind.textContent = 'No legal play — you must pass';
    ind.className = 'must-pass';
  } else {
    ind.textContent = state.turn ? 'Your turn' : 'CPU is thinking…';
    ind.className   = state.turn ? 'your-turn' : 'cpu-turn';
  }

  // Player hand with click handlers
  renderHand(state.player, document.getElementById('player-hand'), false, (i) => {
    if (!state.turn || state.gamePhase !== 'play') return;
    state.player[i].selected = !state.player[i].selected;
    state.invalidMove = false;
    renderBoard();
  });

  // Button states: normal controls during play, a single "see result" in review.
  const playBtn   = document.getElementById('play-btn');
  const passBtn   = document.getElementById('pass-btn');
  const clearBtn  = document.getElementById('clear-btn');
  const rulesBtn  = document.getElementById('rules-btn');
  const resultBtn = document.getElementById('result-btn');
  [playBtn, passBtn, clearBtn, rulesBtn].forEach(b => {
    b.style.display = reviewing ? 'none' : '';
  });
  resultBtn.style.display = reviewing ? '' : 'none';

  playBtn.disabled  = !yourTurn;
  passBtn.disabled  = !yourTurn || state.currType[0] === 0;
  clearBtn.disabled = !yourTurn;
  passBtn.classList.toggle('btn-green', mustPass);
  passBtn.classList.toggle('btn-gray', !mustPass);

  if (state.invalidMove) {
    playBtn.textContent = 'Invalid!';
    playBtn.classList.add('btn-invalid');
  } else {
    playBtn.textContent = 'Play';
    playBtn.classList.remove('btn-invalid');
  }
}

function renderHistory() {
  const list = document.getElementById('history-list');
  if (!list) return;
  list.innerHTML = '';
  state.history.forEach(entry => {
    const div = document.createElement('div');
    div.className = 'history-entry history-' + entry.who;
    div.textContent = entry.text;
    list.appendChild(div);
  });
  list.scrollTop = list.scrollHeight;
}

function renderAll() {
  renderOverlay();
  renderBoard();
  renderHistory();
}

// ─── Overlay / Menus ──────────────────────────────────────────────────────────

function renderOverlay() {
  const overlay = document.getElementById('overlay');
  const content = document.getElementById('overlay-content');

  // Rules overlay takes priority and doesn't change gamePhase
  if (state.showRules) {
    overlay.classList.remove('hidden');
    renderRulesPage(content);
    return;
  }

  // During play and the end-of-hand review, the overlay stays out of the way
  // so the board (and final move) is visible.
  if (state.gamePhase === 'play' || state.gamePhase === 'review') {
    overlay.classList.add('hidden');
    return;
  }

  overlay.classList.remove('hidden');
  content.innerHTML = '';

  if (state.gamePhase === 'menu') {
    const hasScore = state.playerScore + state.cpuScore > 0;
    content.innerHTML = `
      <h1 class="overlay-title">Big 2</h1>
      <p class="overlay-sub">Shanghainese card game — 2 players</p>
      ${hasScore ? `
      <div class="overlay-scores">
        <span>You: <strong>${state.playerScore}</strong> pts</span>
        <span>CPU: <strong>${state.cpuScore}</strong> pts</span>
      </div>` : ''}
      <div class="overlay-btns">
        <button id="start-btn" class="btn btn-green">New Game</button>
        <button id="rules-btn-menu" class="btn btn-indigo">Rules</button>
      </div>
    `;
    document.getElementById('start-btn').addEventListener('click', startGame);
    document.getElementById('rules-btn-menu').addEventListener('click', showRulesOverlay);

  } else if (state.gamePhase === 'gameover') {
    const r = state.gameResult;
    const youWon = r.winner === 'player';
    content.innerHTML = `
      <h1 class="overlay-title">${youWon ? 'You Win!' : 'CPU Wins'}</h1>
      <p class="overlay-sub">${youWon ? 'CPU' : 'You'} had <strong>${r.loserRemaining}</strong> card${r.loserRemaining !== 1 ? 's' : ''} remaining</p>
      <p class="overlay-sub">Points earned: <strong>${r.points}</strong>${r.bonus ? ` (${r.loserRemaining}-card bonus)` : ''}</p>
      <div class="overlay-scores">
        <span>You: <strong>${state.playerScore}</strong> pts</span>
        <span class="series-note">First to 50 pts wins series</span>
        <span>CPU: <strong>${state.cpuScore}</strong> pts</span>
      </div>
      <div class="overlay-btns">
        ${r.seriesWon
          ? `<button id="series-result-btn" class="btn btn-green">Series Result</button>`
          : `<button id="next-game-btn" class="btn btn-green">Next Game</button>
             <button id="new-series-btn" class="btn btn-gray">New Series</button>`}
      </div>
    `;
    if (r.seriesWon) {
      document.getElementById('series-result-btn').addEventListener('click', () => {
        state.gamePhase = 'series-over';
        renderOverlay();
      });
    } else {
      document.getElementById('next-game-btn').addEventListener('click', startGame);
      document.getElementById('new-series-btn').addEventListener('click', newSeries);
    }

  } else if (state.gamePhase === 'series-over') {
    const r = state.gameResult;
    const youWon = r.seriesWinner === 'player';
    content.innerHTML = `
      <h1 class="overlay-title">${youWon ? 'You Win the Series!' : 'CPU Wins the Series'}</h1>
      <div class="overlay-scores">
        <span>You: <strong>${state.playerScore}</strong> pts</span>
        <span>CPU: <strong>${state.cpuScore}</strong> pts</span>
      </div>
      <div class="overlay-btns">
        <button id="new-series-btn2" class="btn btn-green">New Series</button>
      </div>
    `;
    document.getElementById('new-series-btn2').addEventListener('click', newSeries);
  }
}

function renderRulesPage(content) {
  const pages = getRulesPages();
  const page  = pages[state.rulesPage - 1];
  content.innerHTML = `
    <h2 class="overlay-title rules-page-title">${page.title}</h2>
    <div class="rules-text">${page.content}</div>
    <div class="overlay-btns">
      ${state.rulesPage > 1
        ? `<button id="rules-prev" class="btn btn-gray">← Back</button>`
        : ''}
      <button id="rules-close" class="btn btn-gray">${state.gamePhase === 'play' ? 'Close' : 'Back to Menu'}</button>
      ${state.rulesPage < pages.length
        ? `<button id="rules-next" class="btn btn-indigo">Next →</button>`
        : ''}
    </div>
    <p style="text-align:center;font-size:0.78rem;color:#6b7280;margin-top:-4px">
      Page ${state.rulesPage} of ${pages.length}
    </p>
  `;
  document.getElementById('rules-close').addEventListener('click', hideRulesOverlay);
  if (state.rulesPage > 1) {
    document.getElementById('rules-prev').addEventListener('click', () => {
      state.rulesPage--;
      renderRulesPage(content);
    });
  }
  if (state.rulesPage < pages.length) {
    document.getElementById('rules-next').addEventListener('click', () => {
      state.rulesPage++;
      renderRulesPage(content);
    });
  }
}

// ─── Rules Content ────────────────────────────────────────────────────────────
function getRulesPages() {
  return [
    {
      title: 'Overview & Setup',
      content: `
        <p><strong>Goal:</strong> Be the first to play all your cards. In a series, accumulate points by leaving your opponent with more cards.</p>
        <p><strong>Deck:</strong> Standard 52-card deck — remove three 2s and one Ace, leaving one 2 and three Aces. Suits are ignored; only ranks matter.</p>
        <p><strong>Deal:</strong> Shuffle and split into three piles of 16. Each player takes one pile; the third is unused so hands are partly unknown.</p>
        <p><strong>Initiative:</strong> The player with 3♠ goes first. If neither has it, check 4♠, 5♠, … In subsequent games of a series, the previous winner leads.</p>
      `,
    },
    {
      title: 'How to Play',
      content: `
        <p>The player with initiative starts a trick by playing any valid combination. The opponent must play the <strong>same combination type and length at a higher rank</strong>, or pass. If either player passes, the trick ends — the last player to play wins the trick and leads the next one with any combination.</p>
        <h3>Valid Combinations</h3>
        <ul>
          <li><strong>Single:</strong> Any one card. Order: 3 &lt; 4 &lt; … &lt; K &lt; A &lt; 2 (highest).</li>
          <li><strong>Pair:</strong> Two cards of the same rank. No double 2.</li>
          <li><strong>Triple:</strong> Three of the same rank. Triple Ace = Bomb.</li>
          <li><strong>Full House:</strong> Triple + pair of a different rank. Ranked by the triple.</li>
          <li><strong>Straight:</strong> 5+ consecutive singles. A can follow K; 2 can be high (J-Q-K-A-2) or low (2-3-4-5-6). Note: A-2-3-4-5 is <em>invalid</em>.</li>
          <li><strong>Sisters:</strong> 2+ consecutive pairs (e.g. 6-6, 7-7). No double 2. Ranked by highest pair.</li>
          <li><strong>Triple Straight:</strong> 2+ consecutive triples. Ranked by highest triple.</li>
          <li><strong>Bomb:</strong> Four of same rank, or triple Ace (highest bomb). You may add one extra card. Beats any non-bomb; compared by bomb rank. Ace bomb is the highest.</li>
        </ul>
      `,
    },
    {
      title: 'Scoring & Series',
      content: `
        <p>The first player to play all their cards wins the game.</p>
        <p><strong>Points:</strong> The winner scores the number of cards left in the loser's hand. If the loser barely played, a fixed bonus score applies <em>instead</em> (it replaces the count, it doesn't add to it):</p>
        <ul>
          <li>16 cards remaining (never played): 50 points</li>
          <li>15 cards remaining: 40 points</li>
          <li>14 cards remaining: 30 points</li>
          <li>13 cards remaining: 20 points</li>
        </ul>
        <p><strong>Series:</strong> The first player to reach <strong>50 points</strong> wins the series. The winner of each game leads the next.</p>
        <p>Use the <em>New Series</em> button at any time to reset scores and start fresh.</p>
      `,
    },
  ];
}

// ─── Game Flow ────────────────────────────────────────────────────────────────
function showRulesOverlay() {
  state.showRules = true;
  state.rulesPage = 1;
  renderOverlay();
}

function hideRulesOverlay() {
  state.showRules = false;
  renderAll();
}

function startGame() {
  deal();
  state.current     = [];
  state.currType    = [0, 0, 0];
  state.discard     = new Array(13).fill(0);
  state.gamePhase   = 'play';
  state.showRules   = false;
  state.invalidMove = false;
  state.history     = [];
  pushHistory('system', state.turn ? 'New hand — your lead.' : 'New hand — CPU leads.');
  renderAll();
  if (!state.turn) {
    setTimeout(runCpuTurn, 900);
  }
}

function newSeries() {
  state.playerScore = 0;
  state.cpuScore    = 0;
  state.lastWinner  = null;
  state.gameResult  = null;
  state.gamePhase   = 'menu';
  state.showRules   = false;
  state.history     = [];
  renderAll();
}

function runCpuTurn() {
  // Guard: only run when it's actually the CPU's turn during active play
  if (state.gamePhase !== 'play' || state.turn) return;
  cpuChoose();
  renderAll();
}

// ─── Event Handlers ───────────────────────────────────────────────────────────
document.getElementById('play-btn').addEventListener('click', () => {
  if (!state.turn || state.gamePhase !== 'play') return;
  const type = getPlay();
  if (validPlay(type)) {
    state.invalidMove = false;
    playCurrent(true);
    renderAll();
    if (state.gamePhase === 'play') {
      setTimeout(runCpuTurn, 900);
    }
  } else {
    state.invalidMove = true;
    renderBoard();
    setTimeout(() => {
      state.invalidMove = false;
      const btn = document.getElementById('play-btn');
      if (btn) { btn.textContent = 'Play'; btn.classList.remove('btn-invalid'); }
    }, 1200);
  }
});

document.getElementById('pass-btn').addEventListener('click', () => {
  if (!state.turn || state.currType[0] === 0 || state.gamePhase !== 'play') return;
  pushHistory('player', 'You passed');
  state.current  = [];
  state.currType = [0, 0, 0];
  state.turn     = false;
  renderAll();
  setTimeout(runCpuTurn, 900);
});

document.getElementById('clear-btn').addEventListener('click', () => {
  if (state.gamePhase !== 'play') return;
  state.player.forEach(c => { c.selected = false; });
  state.invalidMove = false;
  renderBoard();
});

document.getElementById('rules-btn').addEventListener('click', showRulesOverlay);

document.getElementById('result-btn').addEventListener('click', () => {
  if (state.gamePhase !== 'review') return;
  state.gamePhase = 'gameover';
  renderAll();
});

document.getElementById('history-clear-btn').addEventListener('click', () => {
  state.history = [];
  renderHistory();
});

document.addEventListener('keydown', e => {
  if (state.showRules) return;
  if (state.gamePhase === 'review' && (e.key === ' ' || e.key === 'Enter')) {
    e.preventDefault();
    document.getElementById('result-btn').click();
    return;
  }
  if (state.gamePhase !== 'play' || !state.turn) return;
  if (e.key === ' ' || e.key === 'Enter') {
    e.preventDefault();
    document.getElementById('play-btn').click();
  } else if (e.key === 'p' || e.key === 'P') {
    document.getElementById('pass-btn').click();
  } else if (e.key === 'Escape') {
    document.getElementById('clear-btn').click();
  }
});

// ─── Initialise ───────────────────────────────────────────────────────────────
function setLoadingProgress(frac) {
  const pct = Math.round(frac * 100);
  const bar = document.getElementById('ai-progress-bar');
  const txt = document.getElementById('ai-progress-text');
  if (bar) bar.style.width = pct + '%';
  if (txt) txt.textContent = pct + '%';
}

(async function boot() {
  try {
    await Big2.init(setLoadingProgress);
  } catch (e) {
    const txt = document.getElementById('ai-progress-text');
    if (txt) txt.textContent = 'Failed to load AI: ' + (e && e.message ? e.message : e);
    return;
  }
  const loading = document.getElementById('ai-loading');
  if (loading) loading.classList.add('hidden');
  renderAll();
})();
