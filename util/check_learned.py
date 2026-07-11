from memory.memorize import AikoMemorize

memorize = AikoMemorize(silent=True)
learned = [
    m for m in memorize.get_all()
    if "[self-learned:" in (m.get("memory") or m.get("text") or "")
]
print(f"Found {len(learned)} self-learned entries.\n")
for m in learned:
    text = m.get("memory") or m.get("text") or ""
    print(m.get("created_at"), "-", text[:200])
