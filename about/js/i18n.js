(function () {
  if (window.AboutMultiespectralFusionI18n) return;

  var translations = {
    en: {
      headerTitle: 'About',
      headerSubtitle: 'MultiespectralFusion — Visible + LWIR image fusion',
      aboutSectionTitle: 'About',
      bodyHtml: '<p>This tool is part of a PhD thesis in robotics, computer vision, and artificial intelligence at the <strong>Universidad Miguel Hernández de Elche (UMH)</strong>, Spain. The thesis focuses on multispectral fusion techniques for detection tasks on mobile robots.</p><p>This web demonstrates <strong>static early fusion</strong> methods that combine visible (RGB) and long-wave infrared (LWIR) thermal images at the pixel level, before any feature extraction, showing how each method exploits the complementary information of the two spectra.</p><p>Developed within the <strong>ARVC</strong> (Automation, Robotics and Computer Vision) research group, part of the <strong><a href="https://i3e.umh.es/" target="_blank" rel="noopener noreferrer">I3E</a></strong> research institute at UMH.</p>'
    },
    es: {
      headerTitle: 'Acerca de',
      headerSubtitle: 'MultiespectralFusion — Fusión de imágenes visible + LWIR',
      aboutSectionTitle: 'Acerca de',
      bodyHtml: '<p>Esta herramienta forma parte de una tesis doctoral en robótica, visión por computador e inteligencia artificial en la <strong>Universidad Miguel Hernández de Elche (UMH)</strong>, España. La tesis se centra en técnicas de fusión multiespectral para tareas de detección a bordo de robots móviles.</p><p>Esta web demuestra métodos de <strong>fusión temprana estática</strong> que combinan imágenes visibles (RGB) e infrarrojas de onda larga (LWIR) a nivel de píxel, antes de cualquier extracción de características, mostrando cómo cada método aprovecha la información complementaria de ambos espectros.</p><p>Desarrollado en el grupo de investigación <strong>ARVC</strong> (Automatización, Robótica y Visión por Computador), parte del instituto de investigación <strong><a href="https://i3e.umh.es/" target="_blank" rel="noopener noreferrer">I3E</a></strong> de la UMH.</p>'
    }
  };

  var initialLang = window.SharedUiCore ? window.SharedUiCore.readLangFromUrl('en') : 'en';
  var api = window.SharedI18nCore
    ? window.SharedI18nCore.createI18n(translations, { initialLang: initialLang, fallbackLang: 'en' })
    : {
        getCopy: function (lang) { return translations[lang === 'es' ? 'es' : 'en']; },
        getLang: function () { return initialLang; },
        setLang: function (lang) { initialLang = lang === 'es' ? 'es' : 'en'; return initialLang; },
      };

  window.AboutMultiespectralFusionI18n = api;
})();
