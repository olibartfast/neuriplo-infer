# Plan — NERT default and CI

1. Set the fresh CMake default and setup-script default to NERT; retain explicit
   OpenCV selection and the existing OpenCV version preference.
2. Add NERT to the existing CI build/test matrix using the shared native steps,
   separate caches, no DEFAULT_BACKEND override, and a capabilities assertion.
3. Update README, deployment/build instructions, agent entrypoints, constitution,
   changelog, and roadmap together. Leave explicit backend examples intact.
4. Build/test fresh NERT and explicit OpenCV configurations; validate capabilities,
   generated docs, setup help, workflow syntax, and documentation links.
5. Integrate through a develop PR and wait for green hosted CI before release.
