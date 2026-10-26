#pragma once

#include "help/help_topic_table.h"
#include "help/help_view_model.h"
#include "view/filter_paint.h"
#include "view/filter_row.h"
#include "view/list_item.h"
#include "view/loss_paint.h"
#include "view/sprite_grid_paint.h"

#include <engine/ui/view_model.h>

#include <array>
#include <memory>
#include <string>

namespace game {

// Everything app.xml binds. Behavior lives in DatasetPanel, ModelPanel and TrainPanel; this type holds state, the paints
// and the tabs.
class AppViewModel final : public engine::ui::ViewModel {
public:
    enum class Tab { Train, Models, Generate, Filters, Process, Datasets, Help };

    AppViewModel();

    void show(Tab tab);

    // Window-wide.
    engine::ui::Bindable<bool> tabTrain{true};
    engine::ui::Bindable<bool> tabModels{false};
    engine::ui::Bindable<bool> tabGenerate{false};
    engine::ui::Bindable<bool> tabFilters{false};
    engine::ui::Bindable<bool> tabProcess{false};
    engine::ui::Bindable<bool> tabDatasets{false};
    engine::ui::Bindable<bool> tabHelp{false};
    engine::ui::MethodCommand showTrain;
    engine::ui::MethodCommand showModels;
    engine::ui::MethodCommand showGenerate;
    engine::ui::MethodCommand showFilters;
    engine::ui::MethodCommand showProcess;
    engine::ui::MethodCommand showDatasets;
    engine::ui::MethodCommand showHelp;
    engine::ui::Bindable<std::string> statusText;
    // The language: the one in use is marked.
    engine::ui::Bindable<bool> localeIsEn{true};
    engine::ui::Bindable<bool> localeIsUk{false};
    engine::ui::MethodCommand localeEn;
    engine::ui::MethodCommand localeUk;

    // The Datasets tab. `datasets` is also the list the Models tab picks the dataset of a new model from.
    engine::ui::Bindable<std::string> importFolderText;
    engine::ui::Bindable<std::string> datasetName;
    engine::ui::Bindable<std::string> importText;
    engine::ui::Bindable<bool> importing{false};
    engine::ui::BindableList<std::shared_ptr<ListItem>> datasets;
    engine::ui::Bindable<std::string> datasetInfoText;
    engine::ui::Bindable<std::string> pageText;
    engine::ui::MethodCommand browseFolder;
    engine::ui::MethodCommand importDataset;
    engine::ui::MethodCommand cancelImport;
    engine::ui::MethodCommand deleteDataset;
    engine::ui::MethodCommand pagePrev;
    engine::ui::MethodCommand pageNext;
    SpriteGridPaint datasetPaint;

    // The Models tab. Each size has a button, marked while it is the chosen one.
    engine::ui::Bindable<std::string> newModelName;
    engine::ui::Bindable<std::string> newModelDatasetText;
    engine::ui::Bindable<bool> sizeSmall{false};
    engine::ui::Bindable<bool> sizeMedium{true};
    engine::ui::Bindable<bool> sizeLarge{false};
    engine::ui::Bindable<std::string> newModelSizeText;
    engine::ui::BindableList<std::shared_ptr<ListItem>> models;
    engine::ui::Bindable<std::string> modelInfoText;
    engine::ui::MethodCommand pickSmall;
    engine::ui::MethodCommand pickMedium;
    engine::ui::MethodCommand pickLarge;
    engine::ui::MethodCommand createModel;
    engine::ui::MethodCommand openModel;
    engine::ui::MethodCommand openModelWithDataset;
    engine::ui::MethodCommand deleteModel;

    // The Generate tab: pictures drawn by the model marked in the Models tab.
    engine::ui::Bindable<std::string> generateModelText;
    engine::ui::Bindable<std::string> generateCount{"1000"};
    engine::ui::Bindable<std::string> generateFolderText;
    engine::ui::Bindable<std::string> generateText;
    engine::ui::Bindable<bool> backgroundClear{true};
    engine::ui::Bindable<bool> backgroundKeep{false};
    engine::ui::MethodCommand pickClear;
    engine::ui::MethodCommand pickKeep;
    engine::ui::MethodCommand chooseOutput;
    engine::ui::MethodCommand generate;
    engine::ui::MethodCommand cancelGenerate;
    SpriteGridPaint generatePaint;

    // The model that the Filters and Process tabs look into: marked in the Models tab, then loaded.
    engine::ui::Bindable<std::string> inspectModelText;
    engine::ui::Bindable<bool> inspectLoaded{false};
    engine::ui::Bindable<bool> inspectEmpty{true};
    engine::ui::MethodCommand loadInspect;

    // The Filters tab: one stage of the network at a time as a grid of pictures, one per channel.
    engine::ui::Bindable<std::string> filterLayerText;
    engine::ui::Bindable<std::string> filterNoteText;
    engine::ui::Bindable<std::string> filterTitleText;
    engine::ui::Bindable<std::string> filterDetailText;
    engine::ui::Bindable<std::string> filterProgressText;
    engine::ui::Bindable<std::string> noiseText;
    engine::ui::Bindable<std::string> pictureText;
    engine::ui::Bindable<bool> modeAnswers{true};
    engine::ui::Bindable<bool> modeWeights{false};
    engine::ui::Bindable<bool> modeDream{false};
    engine::ui::BindableList<std::shared_ptr<FilterRow>> filterRows;
    engine::ui::MethodCommand layerUp;
    engine::ui::MethodCommand layerDown;
    engine::ui::MethodCommand showAnswers;
    engine::ui::MethodCommand showWeights;
    engine::ui::MethodCommand showDream;
    engine::ui::MethodCommand noiseLess;
    engine::ui::MethodCommand noiseMore;
    engine::ui::MethodCommand newPicture;
    FilterPaint inputSpritePaint;
    FilterPaint inputNoisyPaint;
    FilterPaint inputEstimatePaint;
    FilterPaint inputNoisePaint;
    FilterPaint detailAnswerPaint;
    FilterPaint detailWeightsPaint;
    FilterPaint detailDreamPaint;

    // The Process tab: how sprites come out of noise, which dataset sprite each is closest to, and a check for copying.
    engine::ui::Bindable<std::string> processText;
    engine::ui::Bindable<std::string> copyText;
    engine::ui::Bindable<bool> viewEstimate{true};
    engine::ui::Bindable<bool> viewNoisy{false};
    engine::ui::Bindable<bool> copyRunning{false};
    engine::ui::MethodCommand newSamples;
    engine::ui::MethodCommand pickEstimate;
    engine::ui::MethodCommand pickNoisy;
    engine::ui::MethodCommand checkCopies;
    engine::ui::MethodCommand cancelCopies;
    SpriteGridPaint processPaint;

    // The Help tab. Topic i is a one-item list so that its document is built from the template; helpShown[i] says
    // which topic is on screen.
    engine::ui::BindableList<std::shared_ptr<HelpNavItem>> helpNav;
    engine::ui::Bindable<float> helpScroll{0.0f};
    std::array<engine::ui::BindableList<std::shared_ptr<HelpTopicViewModel>>, kHelpTopicCount> helpTopic;
    std::array<engine::ui::Bindable<bool>, kHelpTopicCount> helpShown;

    // The Train tab, for the model that is open.
    engine::ui::Bindable<std::string> modelTitleText;
    engine::ui::Bindable<std::string> modelDetailText;
    engine::ui::Bindable<std::string> stateText;
    engine::ui::Bindable<std::string> stepText;
    engine::ui::Bindable<std::string> lossText;
    engine::ui::Bindable<std::string> toggleText{"Start"};
    engine::ui::Bindable<std::string> checkpointText;
    engine::ui::Bindable<std::string> previewText;
    // Start / pause / resume, whichever the run is ready for.
    engine::ui::MethodCommand toggle;
    engine::ui::MethodCommand stop;
    engine::ui::MethodCommand newPreview;
    engine::ui::MethodCommand saveCheckpoint;
    LossPaint lossPaint;
    SpriteGridPaint previewPaint;

private:
    void show_train() { show(Tab::Train); }
    void show_models() { show(Tab::Models); }
    void show_generate() { show(Tab::Generate); }
    void show_filters() { show(Tab::Filters); }
    void show_process() { show(Tab::Process); }
    void show_datasets() { show(Tab::Datasets); }
    void show_help() { show(Tab::Help); }
};

}
