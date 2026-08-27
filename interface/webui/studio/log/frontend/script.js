const apiBase = `${window.location.pathname.replace(/\/$/, '')}/api`;
const form = document.querySelector('#filters');
const status = document.querySelector('#status');
const entries = document.querySelector('#entries');
const template = document.querySelector('#entry-template');
let refreshGeneration = 0;

function addOptions(id, values) {
  const select = document.querySelector(id);
  select.replaceChildren(...values.map(value => new Option(value, value)));
}

async function loadOptions() {
  const response = await fetch(`${apiBase}/options`);
  if (!response.ok) throw new Error('Could not load log filters');
  const data = await response.json();
  addOptions('#levels', data.levels);
  addOptions('#components', data.components);
}

async function loadEntries(generation) {
  const params = new URLSearchParams(new FormData(form));
  [...params.entries()].filter(([, value]) => !value).forEach(([key]) => params.delete(key));
  status.textContent = 'Loading log…';
  const response = await fetch(`${apiBase}/entries?${params}`);
  const data = await response.json();
  if (generation !== refreshGeneration) return;
  if (!response.ok) throw new Error(data.detail || 'Could not load log entries');
  entries.replaceChildren(...data.entries.map(entry => {
    const fragment = template.content.cloneNode(true);
    const article = fragment.querySelector('article');
    article.classList.add(`level-${entry.level.toLowerCase()}`);
    fragment.querySelector('time').textContent = entry.timestamp;
    fragment.querySelector('.level').textContent = entry.level;
    fragment.querySelector('.component').textContent = entry.component;
    fragment.querySelector('pre').textContent = entry.message;
    return fragment;
  }));
  status.textContent = `${data.count} shown of ${data.total} entries`;
}

async function refresh(loadFilters = false) {
  const generation = ++refreshGeneration;
  try {
    if (loadFilters) await loadOptions();
    await loadEntries(generation);
  }
  catch (error) {
    if (generation === refreshGeneration) status.textContent = error.message;
  }
}
form.addEventListener('submit', event => { event.preventDefault(); refresh(); });
form.addEventListener('reset', () => setTimeout(refresh));
document.querySelector('#refresh').addEventListener('click', () => refresh());
refresh(true);
