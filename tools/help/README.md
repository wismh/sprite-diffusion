# Help topics and messages

`help/help_content.py` holds the text of the Help tab in English and Ukrainian. `strings_ui.py` holds the texts of the
window (`assets/ui/app.xml`), and `strings_msg.py` the messages the program writes while it works. Run, after a change
of any of them:

```
python tools/help/gen_help.py
python tools/gen_locale.py
```

`gen_help.py` writes `assets/ui/help_topics/*.xml`, `src/help/help_topic_table.h` and the list of topics in `app.xml`.
`gen_locale.py` writes `assets/locale/en.strings` and `uk.strings` from all three tables, and stops when the code or a
document uses a key that has no text.
