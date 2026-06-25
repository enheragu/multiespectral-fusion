---
---
(function () {
  // tools_base resolves to the dev path (/multiespectral_fusion_web) or the prod
  // baseurl (/multiespectral-fusion) at build time, so the menu links work in both.
  var toolsBase = '{{ site.tools_base | default: site.baseurl }}';

  var baseConfig = {
    showBackButton: false,
    homePath: toolsBase + '/',
    currentPath: toolsBase + '/',
    preserveLangParam: true,
    menuSections: [
      {
        items: [
          { href: toolsBase + '/', label: { en: 'Multiespectral Fusion', es: 'Multiespectral Fusion' } }
        ]
      },
      {
        items: [
          { href: toolsBase + '/about/', label: { en: 'About', es: 'Acerca de' } }
        ]
      }
    ]
  };

  var pageConfig = window.StatToolsNavPageConfig || {};
  var resolved = Object.assign({}, baseConfig, pageConfig);

  if (!Object.prototype.hasOwnProperty.call(pageConfig, 'menuSections')) {
    resolved.menuSections = baseConfig.menuSections;
  }

  window.ToolsNavConfig = resolved;
})();
