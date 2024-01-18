

# Release checklist
- Audit import paths, make sure everything is organized in a sensible fashion.
- Complete the lib export directory
- Make sure names are consistent. Module -> Metric most likely
- Add `pre_init` methods to all of the modules that can download models so we can force downloads upfront. These should probably be called
  from the library creator methods.
