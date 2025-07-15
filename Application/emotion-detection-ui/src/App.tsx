import React, { useState } from 'react';
import './App.css';
import CheckBoxPanel from './components/CheckBoxPanel';
import EmotionResult from './components/EmotionResult';
import LeftPanel from './components/LeftPanel';

const App: React.FC = () => {
  const [emotionResults, setEmotionResults] = useState<{ [key: string]: number }>({});
  const [selectedModel, setSelectedModel] = useState<string>('vit_b16.pth');

  const handleResult = (results: { [key: string]: number }) => {
    setEmotionResults(results);
  };

  return (
    <div className="app-container">
      {/* Left Panel (2/3 width) */}
      <div className="left-panel">
        <LeftPanel onResult={handleResult} selectedModel={selectedModel} />
      </div>

      {/* Right Panel (1/3 width) */}
      <div className="right-panel">
        <div className="project-title">
          <h2>Emotion Detection Demo</h2>
        </div>
        <CheckBoxPanel />
        <div className="model-selector" style={{ marginBottom: '1rem' }}>
          <h4>Select Model</h4>
          <label>
            <input
              type="radio"
              value="vit_b16.pth"
              checked={selectedModel === 'vit_b16.pth'}
              onChange={(e) => setSelectedModel(e.target.value)}
            />
            VIT B16
          </label>
          <br />
          <label>
            <input
              type="radio"
              value="TransNeXt-B.pth"
              checked={selectedModel === 'TransNeXt-B.pth'}
              onChange={(e) => setSelectedModel(e.target.value)}
            />
            TransNeXt-B
          </label>
          <br />
          <label>
            <input
              type="radio"
              value="ShuffleViT-DFER.t7"
              checked={selectedModel === 'ShuffleViT-DFER.t7'}
              onChange={(e) => setSelectedModel(e.target.value)}
            />
            ShuffleViT-DFER
          </label>
        </div>
        <EmotionResult emotionResults={emotionResults} />
      </div>
    </div>
  );
};

export default App;
