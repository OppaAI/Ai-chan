# AI Assistant Skills

---

## Skill 1: Deep Research

**Triggers:** "research", "deep dive", "explain in depth", "tell me everything about", "what is the full story on"

**Procedure:**

1. Call `web_search` with the topic to get an overview and find top sources.
2. Pick the 2 most relevant URLs from the search results.
3. Call `fetch_page` on both URLs to read the full text.
4. Combine the information. Ignore irrelevant details.
5. Call `final_answer` with a structured report containing:
   - **Overview:** A short summary of the topic.
   - **Key Facts:** 3–4 bullet points of the most important information.
   - **Sources:** A list of the URLs you read.

---

## Skill 2: Fact-Checking

**Triggers:** "is it true that", "fact check", "verify", "did X really happen", "is this a scam"

**Procedure:**

1. Call `web_search` with the claim to see what sources say.
2. Call `fetch_page` on at least 2 different URLs to verify the claim across multiple sources.
3. Evaluate the evidence.
4. Call `final_answer` stating clearly:
   - **Verdict:** TRUE, FALSE, or MIXED.
   - **Evidence:** What the sources actually said.
   - **Sources:** The URLs used to verify.

---

## Skill 3: Comparisons

**Triggers:** "compare", "vs", "versus", "which is better", "difference between"

**Procedure:**

1. Call `web_search` for the first option to get its specs/pros/cons.
2. Call `web_search` for the second option to get its specs/pros/cons.
3. If the search snippets don't have enough detail, call `fetch_page` on a review page.
4. Call `final_answer` with:
   - A markdown table comparing the two options side-by-side.
   - A recommendation based on the user's likely needs.

---

## Skill 4: Latest News & Current Events

**Triggers:** "latest", "news", "current", "what happened with", "update on"

**Procedure:**

1. Call `web_search` with the topic (the search engine will prioritize recent results).
2. Call `fetch_page` on the top news article URL to get the full story, not just the headline.
3. Call `final_answer` with:
   - A 2–3 sentence summary of the latest developments.
   - The source URL.

---

## General Rules for Tool Use

- If a `web_search` result has a snippet that fully answers the question, you do not always need to call `fetch_page`. Just go straight to `final_answer`.
- If a `fetch_page` fails or returns an error, try fetching a different URL from the search results.
- Never invent URLs. Only use URLs returned by `web_search`.
- Always use `final_answer` to deliver the response to the user. Do not just output text without calling the tool.
