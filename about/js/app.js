(function () {
  if (window.AboutMultiespectralFusionApp) return;

  function init() {
    if (!window.SharedToolPageShell || typeof window.SharedToolPageShell.initToolPage !== 'function') {
      document.documentElement.classList.remove('i18n-pending');
      return;
    }

    window.SharedToolPageShell.initToolPage({
      toolTitle: 'About — MultiespectralFusion',
      i18nApi: window.AboutMultiespectralFusionI18n,
      onApplyLanguage: function (copy, lang, setText, setHtml) {
        if (!copy) return;
        setText('about-section-title', copy.aboutSectionTitle);
        setHtml('about-body-text', copy.bodyHtml);
      }
    });
  }

  window.AboutMultiespectralFusionApp = { init: init };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
