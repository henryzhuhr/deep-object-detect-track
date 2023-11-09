import { defineClientConfig } from '@vuepress/client'

import "katex/dist/katex.min.css";
import "./styles/katex.css";

export default defineClientConfig({
  enhance({ app, router, siteData }) {},
  setup() {},
  rootComponents: [],
})