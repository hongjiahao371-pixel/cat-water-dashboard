(function () {
  const key = 'catWaterApiToken';
  const params = new URLSearchParams(location.search);
  const incoming = params.get('token');
  if (incoming) {
    localStorage.setItem(key, incoming);
    params.delete('token');
    const query = params.toString();
    history.replaceState(null, '', location.pathname + (query ? '?' + query : '') + location.hash);
  }

  const nativeFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const request = input instanceof Request ? input : null;
    const url = new URL(request ? request.url : String(input), location.href);
    const options = Object.assign({}, init || {});
    const token = localStorage.getItem(key) || '';
    if (token && url.origin === location.origin && url.pathname.startsWith('/api/')) {
      const headers = new Headers(request ? request.headers : undefined);
      new Headers(options.headers || {}).forEach((value, name) => headers.set(name, value));
      headers.set('X-API-Token', token);
      options.headers = headers;
    }
    return nativeFetch(input, options);
  };
})();
