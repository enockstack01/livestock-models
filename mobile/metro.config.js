// Bundle .tflite models as assets (react-native-fast-tflite loads them with require()).
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('tflite');
module.exports = config;
