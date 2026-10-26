#include "view/app_view_model.h"

namespace game {

AppViewModel::AppViewModel() {
    using engine::ui::intern;

    property(intern("tabTrain"), tabTrain);
    property(intern("tabModels"), tabModels);
    property(intern("tabGenerate"), tabGenerate);
    property(intern("tabFilters"), tabFilters);
    property(intern("tabProcess"), tabProcess);
    property(intern("tabDatasets"), tabDatasets);
    property(intern("tabHelp"), tabHelp);
    command(intern("showTrain"), showTrain);
    command(intern("showModels"), showModels);
    command(intern("showGenerate"), showGenerate);
    command(intern("showFilters"), showFilters);
    command(intern("showProcess"), showProcess);
    command(intern("showDatasets"), showDatasets);
    command(intern("showHelp"), showHelp);
    showTrain.bind_to<AppViewModel, &AppViewModel::show_train>(*this);
    showModels.bind_to<AppViewModel, &AppViewModel::show_models>(*this);
    showGenerate.bind_to<AppViewModel, &AppViewModel::show_generate>(*this);
    showFilters.bind_to<AppViewModel, &AppViewModel::show_filters>(*this);
    showProcess.bind_to<AppViewModel, &AppViewModel::show_process>(*this);
    showDatasets.bind_to<AppViewModel, &AppViewModel::show_datasets>(*this);
    showHelp.bind_to<AppViewModel, &AppViewModel::show_help>(*this);
    property(intern("statusText"), statusText);
    property(intern("localeIsEn"), localeIsEn);
    property(intern("localeIsUk"), localeIsUk);
    command(intern("localeEn"), localeEn);
    command(intern("localeUk"), localeUk);

    property(intern("importFolderText"), importFolderText);
    property(intern("datasetName"), datasetName);
    property(intern("importText"), importText);
    property(intern("importing"), importing);
    property(intern("datasets"), datasets);
    property(intern("datasetInfoText"), datasetInfoText);
    property(intern("pageText"), pageText);
    command(intern("browseFolder"), browseFolder);
    command(intern("importDataset"), importDataset);
    command(intern("cancelImport"), cancelImport);
    command(intern("deleteDataset"), deleteDataset);
    command(intern("pagePrev"), pagePrev);
    command(intern("pageNext"), pageNext);
    paint(intern("datasetPaint"), datasetPaint);

    property(intern("newModelName"), newModelName);
    property(intern("newModelDatasetText"), newModelDatasetText);
    property(intern("sizeSmall"), sizeSmall);
    property(intern("sizeMedium"), sizeMedium);
    property(intern("sizeLarge"), sizeLarge);
    property(intern("newModelSizeText"), newModelSizeText);
    property(intern("models"), models);
    property(intern("modelInfoText"), modelInfoText);
    command(intern("pickSmall"), pickSmall);
    command(intern("pickMedium"), pickMedium);
    command(intern("pickLarge"), pickLarge);
    command(intern("createModel"), createModel);
    command(intern("openModel"), openModel);
    command(intern("openModelWithDataset"), openModelWithDataset);
    command(intern("deleteModel"), deleteModel);

    property(intern("generateModelText"), generateModelText);
    property(intern("generateCount"), generateCount);
    property(intern("generateFolderText"), generateFolderText);
    property(intern("generateText"), generateText);
    property(intern("backgroundClear"), backgroundClear);
    property(intern("backgroundKeep"), backgroundKeep);
    command(intern("pickClear"), pickClear);
    command(intern("pickKeep"), pickKeep);
    command(intern("chooseOutput"), chooseOutput);
    command(intern("generate"), generate);
    command(intern("cancelGenerate"), cancelGenerate);
    paint(intern("generatePaint"), generatePaint);

    property(intern("inspectModelText"), inspectModelText);
    property(intern("inspectLoaded"), inspectLoaded);
    property(intern("inspectEmpty"), inspectEmpty);
    command(intern("loadInspect"), loadInspect);

    property(intern("filterLayerText"), filterLayerText);
    property(intern("filterNoteText"), filterNoteText);
    property(intern("filterTitleText"), filterTitleText);
    property(intern("filterDetailText"), filterDetailText);
    property(intern("filterProgressText"), filterProgressText);
    property(intern("noiseText"), noiseText);
    property(intern("pictureText"), pictureText);
    property(intern("modeAnswers"), modeAnswers);
    property(intern("modeWeights"), modeWeights);
    property(intern("modeDream"), modeDream);
    property(intern("filterRows"), filterRows);
    command(intern("layerUp"), layerUp);
    command(intern("layerDown"), layerDown);
    command(intern("showAnswers"), showAnswers);
    command(intern("showWeights"), showWeights);
    command(intern("showDream"), showDream);
    command(intern("noiseLess"), noiseLess);
    command(intern("noiseMore"), noiseMore);
    command(intern("newPicture"), newPicture);
    paint(intern("inputSpritePaint"), inputSpritePaint);
    paint(intern("inputNoisyPaint"), inputNoisyPaint);
    paint(intern("inputEstimatePaint"), inputEstimatePaint);
    paint(intern("inputNoisePaint"), inputNoisePaint);
    paint(intern("detailAnswerPaint"), detailAnswerPaint);
    paint(intern("detailWeightsPaint"), detailWeightsPaint);
    paint(intern("detailDreamPaint"), detailDreamPaint);

    property(intern("processText"), processText);
    property(intern("copyText"), copyText);
    property(intern("viewEstimate"), viewEstimate);
    property(intern("viewNoisy"), viewNoisy);
    property(intern("copyRunning"), copyRunning);
    command(intern("newSamples"), newSamples);
    command(intern("pickEstimate"), pickEstimate);
    command(intern("pickNoisy"), pickNoisy);
    command(intern("checkCopies"), checkCopies);
    command(intern("cancelCopies"), cancelCopies);
    paint(intern("processPaint"), processPaint);

    property(intern("helpNav"), helpNav);
    property(intern("helpScroll"), helpScroll);
    for (std::size_t i = 0; i < kHelpTopicCount; ++i) {
        property(intern("helpTopic" + std::to_string(i)), helpTopic[i]);
        property(intern("helpShown" + std::to_string(i)), helpShown[i]);
    }

    property(intern("modelTitleText"), modelTitleText);
    property(intern("modelDetailText"), modelDetailText);
    property(intern("stateText"), stateText);
    property(intern("stepText"), stepText);
    property(intern("lossText"), lossText);
    property(intern("toggleText"), toggleText);
    property(intern("checkpointText"), checkpointText);
    property(intern("previewText"), previewText);
    command(intern("toggle"), toggle);
    command(intern("stop"), stop);
    command(intern("newPreview"), newPreview);
    command(intern("saveCheckpoint"), saveCheckpoint);
    paint(intern("lossPaint"), lossPaint);
    paint(intern("previewPaint"), previewPaint);
}

void AppViewModel::show(Tab tab) {
    tabTrain = tab == Tab::Train;
    tabModels = tab == Tab::Models;
    tabGenerate = tab == Tab::Generate;
    tabFilters = tab == Tab::Filters;
    tabProcess = tab == Tab::Process;
    tabDatasets = tab == Tab::Datasets;
    tabHelp = tab == Tab::Help;
}

}
