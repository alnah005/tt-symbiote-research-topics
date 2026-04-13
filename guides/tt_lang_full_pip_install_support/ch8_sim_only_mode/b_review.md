## Pass 1

1. **Broken link: "Guide Index" in `design_options.md` navigation footer.**
   Line 269: `Return to [Guide Index](../index.md)` -- the file `../index.md` does not exist. The only markdown file at the guide root is `plan.md`. Either the link target should be `../plan.md`, or a `../index.md` file needs to be created.

2. **Missing `Prev` navigation link in `design_options.md`.**
   `index.md` ends with `**Next:** [design_options.md](./design_options.md)`, but `design_options.md` has no corresponding `**Prev:** [index.md](./index.md)` footer. Other chapters in this guide use bidirectional navigation; this page should too.

3. **`index.md` lacks a `Prev` link back to Chapter 7.**
   The chapter opens with references to Chapters 6 and 7 but has no `**Prev:** [Chapter 7](../ch7_wheel_packaging/index.md)` navigation footer. The `**Next:** design_options.md` link is present, but there is no way to navigate backward to the preceding chapter.

## Pass 2

All three Pass 1 issues (broken Guide Index link, missing Prev in `design_options.md`, missing Prev to Ch7 in `index.md`) remain unfixed.

1. **Incorrect PEP reference in `design_options.md` line 217.**
   The text cites "implicit namespace packages per PEP 420" but PEP 420 ("Implicit Namespace Packages") was **Rejected** in April 2012. The implicit namespace package mechanism that shipped in Python 3.3 was implemented without a dedicated accepted PEP -- it was part of the importlib-based import system rewrite. The reference should either drop the PEP number entirely (e.g., "using implicit namespace packages as supported since Python 3.3") or cite the relevant section of the Python documentation on namespace packages rather than a rejected PEP.

2. **`design_options.md` line 217: multiple distributions sharing a namespace package requires care not mentioned.**
   The text states that `ttl-sim` and `ttl` can both contribute to the `ttl` namespace and "this works cleanly." In practice, if the full `ttl` package ships a `ttl/__init__.py` (which it likely will, since it is the main package), then `ttl` is a regular package, not a namespace package, and a second distribution (`ttl-sim`) cannot contribute additional sub-packages to it. For this to work, either (a) the full `ttl` wheel must depend on `ttl-sim` and not duplicate those sub-packages, or (b) `ttl/__init__.py` must be omitted from both distributions so that `ttl` is a true implicit namespace package. The chapter's step 5 ("Make `ttl` depend on `ttl-sim`") hints at option (a) but the namespace package claim on line 217 is misleading without this caveat.

3. **Pass 1 navigation issues still present (items 1-3 above).** All three broken/missing navigation links from Pass 1 remain unaddressed.
