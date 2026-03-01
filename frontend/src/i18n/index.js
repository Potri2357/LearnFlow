import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from './locales/en';
import hi from './locales/hi';
import ta from './locales/ta';
import fr from './locales/fr';

i18n
  .use(initReactI18next)
  .init({
    resources: { en: { translation: en }, hi: { translation: hi }, ta: { translation: ta }, fr: { translation: fr } },
    lng: localStorage.getItem('learnflow_lang') || 'en',
    fallbackLng: 'en',
    interpolation: { escapeValue: false },
  });

export default i18n;
