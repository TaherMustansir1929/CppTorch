import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SLIDES,
  AUTHORS,
  UNIVERSITY_NAME,
  CLASS_SECTION,
  DOCS_SITE_URL,
  GITHUB_REPO_URL,
} from './slides';

test('slide deck contains all 13 ordered slides with required fields', () => {
  assert.equal(SLIDES.length, 13);
  SLIDES.forEach((slide, idx) => {
    assert.equal(slide.id, idx + 1);
    assert.ok(slide.title.length > 0);
    assert.ok(slide.subtitle.length > 0);
    assert.ok(slide.bigIdea.length > 0);
    assert.ok(slide.transitionNote.length > 0);
    assert.ok(slide.interactionHint.length > 0);
  });
});

test('authors list matches Taher, Kashif, and Yahya with roll numbers and roles', () => {
  assert.equal(AUTHORS.length, 3);
  assert.deepEqual(
    AUTHORS.map((a) => `${a.name} — ${a.roll} (${a.role})`),
    [
      'Taher — 25K-0119 (Team Lead)',
      'Kashif — 25K-0018 (Feature Engineer)',
      'Yahya — 25K-0078 (Foundational Developer)',
    ]
  );
  assert.equal(UNIVERSITY_NAME, 'FAST NUCES');
  assert.equal(CLASS_SECTION, 'BAI-3C');
  assert.equal(DOCS_SITE_URL, 'https://cpptorch.vercel.app');
  assert.equal(GITHUB_REPO_URL, 'https://github.com/cpptorch/cpptorch');
});
