const fs = require('fs');
const path = require('path');

// 載入 presets.js，會在 global 上註冊 TOY_MODELS
require('./models/presets.js');

const models = global.TOY_MODELS;

// 保留 4 位小數，確保精度同時將檔案精簡
function roundFloats(obj) {
  if (typeof obj === 'number') {
    return Math.round(obj * 10000) / 10000;
  }
  if (Array.isArray(obj)) {
    return obj.map(roundFloats);
  }
  if (obj && typeof obj === 'object') {
    const res = {};
    for (const k of Object.keys(obj)) {
      res[k] = roundFloats(obj[k]);
    }
    return res;
  }
  return obj;
}

const cleaned = roundFloats(models);

const outputContent = `/**
 * Toy Llama Stepper - 預先序列化靜態模型權重 (Static Weights)
 * 供網頁開箱即用、無須執行期計算
 */
window.TOY_MODELS = ${JSON.stringify(cleaned, null, 2)};
`;

fs.writeFileSync(path.join(__dirname, 'models', 'models-static.js'), outputContent, 'utf-8');
console.log('Successfully generated models/models-static.js');
