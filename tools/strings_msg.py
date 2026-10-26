# The messages the program writes while it works (status line, lists, texts that follow the state), in English and
# Ukrainian: key -> (en, uk). `{name}` places are filled by the program.
MSG = {
    "status.ready": ("Ready", "Готово до роботи"),

    # Datasets
    "ds.no_folder": ("No folder chosen", "Теку не вибрано"),
    "ds.no_store": ("There is no folder for datasets: nothing can be imported",
                    "Немає теки для датасетів: нічого не можна імпортувати"),
    "ds.exists": ("A dataset named \"{name}\" exists: choose another name",
                  "Датасет із назвою «{name}» уже є: виберіть іншу назву"),
    "ds.reading_folder": ("Reading the folder...", "Читання теки..."),
    "ds.progress": ("Reading {done} of {total}", "Прочитано {done} з {total}"),
    "ds.cancelled": ("Import cancelled", "Імпорт скасовано"),
    "ds.failed": ("Import failed: {detail}", "Імпорт не вдався: {detail}"),
    "ds.save_failed": ("Cannot save the dataset: {detail}", "Не вдалося зберегти датасет: {detail}"),
    "ds.imported": ("Imported {count} sprites", "Імпортовано спрайтів: {count}"),
    "ds.imported_skipped": ("Imported {count} sprites, {skipped} files skipped (not 16x16 pictures)",
                            "Імпортовано спрайтів: {count}, пропущено файлів: {skipped} (не картинки 16x16)"),
    "ds.added": ("Dataset \"{name}\" added", "Датасет «{name}» додано"),
    "ds.removed": ("Dataset \"{name}\" removed. Models that learn from it will not open",
                   "Датасет «{name}» видалено. Моделі, що на ньому вчаться, не відкриються"),
    "ds.row": ("{name}  ({count} sprites)", "{name}  (спрайтів: {count})"),
    "ds.info": ("{name}: {count} sprites from {source}", "{name}: спрайтів {count}, з {source}"),
    "ds.read_failed": ("Cannot read the dataset: {detail}", "Не вдалося прочитати датасет: {detail}"),
    "ds.none_yet": ("No datasets yet: import a folder of 16x16 pictures",
                    "Датасетів ще немає: імпортуйте теку з картинками 16x16"),
    "ds.choose": ("Choose a dataset", "Виберіть датасет"),
    "ds.page": ("Page {page} of {pages}", "Сторінка {page} з {pages}"),

    # Models
    "model.small": ("Small: 32 channels, quick to train, plain sprites",
                    "Мала: 32 канали, швидко вчиться, прості спрайти"),
    "model.medium": ("Medium: 64 channels, a good start", "Середня: 64 канали, добрий початок"),
    "model.large": ("Large: 96 channels, slow, the most detail", "Велика: 96 каналів, повільна, найбільше деталей"),
    "model.exists": ("A model named \"{name}\" exists: choose another name",
                     "Модель із назвою «{name}» уже є: виберіть іншу назву"),
    "model.cannot_make": ("Cannot make \"{name}\": {detail}", "Не вдалося створити «{name}»: {detail}"),
    "model.deleted": ("Model \"{name}\" deleted", "Модель «{name}» видалено"),
    "model.row": ("{name}  ({base} channels, {steps} steps)", "{name}  (каналів: {base}, кроків: {steps})"),
    "model.info": ("{name}: {base} channels, {steps} steps, dataset \"{dataset}\", {size} MB",
                   "{name}: каналів {base}, кроків {steps}, датасет «{dataset}», {size} МБ"),
    "model.none_yet": ("No models yet: make one on the left", "Моделей ще немає: створіть її ліворуч"),
    "model.choose": ("Choose a model", "Виберіть модель"),
    "model.pick_dataset": ("Choose the dataset below", "Виберіть датасет нижче"),
    "model.dataset_is": ("Dataset: \"{name}\"", "Датасет: «{name}»"),

    # Train
    "train.state.training": ("Training", "Навчання йде"),
    "train.state.paused": ("Paused", "Пауза"),
    "train.state.finished": ("Finished", "Завершено"),
    "train.state.failed": ("Failed", "Збій"),
    "train.state.stopped": ("Stopped", "Зупинено"),
    "train.cannot_open": ("Cannot open \"{name}\": {detail}", "Не вдалося відкрити «{name}»: {detail}"),
    "train.opened": ("Model \"{name}\" is open", "Модель «{name}» відкрито"),
    "train.none": ("No model is open", "Жодну модель не відкрито"),
    "train.none_hint": ("Make a model or open a saved one in the Models tab.",
                        "Створіть модель або відкрийте збережену на вкладці «Моделі»."),
    "train.detail": ("Dataset \"{dataset}\": {count} sprites. Network of {base} channels, {params}M parameters",
                     "Датасет «{dataset}»: спрайтів {count}. Мережа з {base} каналів, параметрів {params} млн"),
    "train.step": ("Step {step}", "Крок {step}"),
    "train.loss": ("Loss {loss}", "Втрати {loss}"),
    "train.loss_none": ("Loss -", "Втрати -"),
    "train.start": ("Start", "Старт"),
    "train.pause": ("Pause", "Пауза"),
    "train.resume": ("Resume", "Продовжити"),
    "train.checkpoint": ("Checkpoint: {path}  (saved {count} times)", "Контрольна точка: {path}  (збережено разів: {count})"),
    "train.samples_at": ("Samples at step {step}", "Зразки на кроці {step}"),
    "train.saved": ("Model \"{name}\" saved", "Модель «{name}» збережено"),
    "train.save_failed": ("Cannot save the model: {detail}", "Не вдалося зберегти модель: {detail}"),
    "train.chart_empty": ("No data yet", "Даних ще немає"),
    "train.failed": ("Failed: {detail}", "Збій: {detail}"),

    # Generate
    "gen.no_folder": ("No folder chosen", "Теку не вибрано"),
    "gen.drawing": ("\"{name}\" is drawing", "«{name}» малює"),
    "gen.mark_model": ("Mark a model in the Models tab", "Позначте модель на вкладці «Моделі»"),
    "gen.model": ("\"{name}\"", "«{name}»"),
    "gen.loading": ("Loading the model...", "Завантаження моделі..."),
    "gen.progress": ("{done} of {total}", "{done} з {total}"),
    "gen.done": ("{count} pictures written to\n{folder}", "Записано картинок: {count}\n{folder}"),
    "gen.cancelled": ("Cancelled: {count} pictures were written to\n{folder}",
                      "Скасовано: записано картинок {count}\n{folder}"),
    "gen.failed": ("Failed: {detail}", "Збій: {detail}"),
    "gen.saved": ("{count} pictures of \"{name}\" saved", "Картинок «{name}» збережено: {count}"),

    # Filters and Process: loading a model
    "insp.hint": ("Mark a model in the Models tab and load it. The weights are read as last saved.",
                  "Позначте модель на вкладці «Моделі» й завантажте її. Ваги читаються так, як були збережені востаннє."),
    "insp.loaded": ("\"{name}\": {steps} steps, dataset \"{dataset}\" ({count} sprites)",
                    "«{name}»: кроків {steps}, датасет «{dataset}» (спрайтів: {count})"),
    "insp.cannot_load": ("Cannot load \"{name}\": {detail}", "Не вдалося завантажити «{name}»: {detail}"),
    "insp.loaded_status": ("Model \"{name}\" loaded", "Модель «{name}» завантажено"),

    # Filters
    "stage.0": ("Input convolution", "Вхідна згортка"),
    "stage.1": ("Block 1 at 16x16", "Блок 1 на 16x16"),
    "stage.2": ("Down to 8x8", "Униз до 8x8"),
    "stage.3": ("Block 2 at 8x8", "Блок 2 на 8x8"),
    "stage.4": ("Down to 4x4", "Униз до 4x4"),
    "stage.5": ("Middle block 1 at 4x4", "Середній блок 1 на 4x4"),
    "stage.6": ("Middle block 2 at 4x4", "Середній блок 2 на 4x4"),
    "stage.7": ("Up to 8x8", "Угору до 8x8"),
    "stage.8": ("Block 3 at 8x8", "Блок 3 на 8x8"),
    "stage.9": ("Up to 16x16", "Угору до 16x16"),
    "stage.10": ("Block 4 at 16x16", "Блок 4 на 16x16"),
    "filters.layer": ("{n} of {total}: {name}  ({channels} channels of {size}x{size})",
                      "{n} з {total}: {name}  (каналів: {channels}, {size}x{size})"),
    "filters.note_answers": (
        "How strongly each channel reacts to the sprite below, as a map (red: more, blue: less). The bar is how much it reacts in all.",
        "Наскільки сильно кожен канал реагує на спрайт унизу, у вигляді карти (червоне: більше, синє: менше). Смужка це загальна сила реакції."),
    "filters.note_weights": (
        "What each channel looks for: its 3x3 weights for every channel before it (red: adds, blue: takes away). The bar is how strong the weights are.",
        "Що шукає кожен канал: його ваги 3x3 для кожного попереднього каналу (червоне: додає, синє: віднімає). Смужка це сила ваг."),
    "filters.note_dream": (
        "The sprite each channel answers to hardest, found by pushing noise toward a stronger answer at this noise level.",
        "Спрайт, на який кожен канал відповідає найсильніше: його знаходять, штовхаючи шум до сильнішої відповіді на цьому рівні шуму."),
    "filters.dream_failed": ("Dreaming failed: {detail}", "Сон не вдався: {detail}"),
    "filters.dream_progress": ("Dreams: {done} of {total}", "Снів: {done} з {total}"),
    "filters.channel": ("Channel {n} of {total}", "Канал {n} з {total}"),
    "filters.channel_detail": (
        "Answers on average {answer}% of the strongest channel; its weights are {weights}% as strong",
        "Відповідає в середньому на {answer}% від найсильнішого каналу; його ваги сильні на {weights}%"),
    "filters.noise_level": ("Noise level {t} of {total}", "Рівень шуму {t} з {total}"),
    "filters.picture": ("Sprite {n} of the dataset", "Спрайт {n} датасету"),

    # Process
    "process.row": ("{n}: closest #{index}, distance {distance}", "{n}: найближчий №{index}, відстань {distance}"),
    "process.row_turned": ("{n}: closest #{index} turned, distance {distance}",
                           "{n}: найближчий №{index} віддзеркалений, відстань {distance}"),
    "process.near_data": (", as close as data is to data", ", так близько, як дані одне до одного"),
    "process.measuring": ("Measuring the dataset...", "Вимірювання датасету..."),
    "process.checking": ("Checking: {percent}%", "Перевірка: {percent}%"),
    "process.cancelled": ("Cancelled", "Скасовано"),
    "process.failed": ("The check failed: {detail}", "Перевірка не вдалася: {detail}"),
    "process.result": (
        "{share}% of {samples} new sprites are as close to a dataset sprite as a typical dataset sprite is to its closest "
        "neighbour ({typical}); about 50% if the new ones sit among the data as the data does.\n"
        "Distance to the closest dataset sprite (10% / 50% / 90%):\nnew sprites {new_low} / {new_median} / {new_high}\n"
        "dataset sprites {data_low} / {data_median} / {data_high}",
        "{share}% із {samples} нових спрайтів так само близькі до спрайта датасету, як типовий спрайт датасету до свого "
        "найближчого сусіда ({typical}); близько 50%, якщо нові стоять серед даних так, як стоять самі дані.\n"
        "Відстань до найближчого спрайта датасету (10% / 50% / 90%):\nнові спрайти {new_low} / {new_median} / {new_high}\n"
        "спрайти датасету {data_low} / {data_median} / {data_high}"),
}
