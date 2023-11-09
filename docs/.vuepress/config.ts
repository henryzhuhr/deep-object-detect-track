import { defineUserConfig } from 'vuepress';
import { defaultTheme } from '@vuepress/theme-default';
import { MarkdownOptions } from '@vuepress/markdown'

const markdownOptions: MarkdownOptions = {
  headers: {
    level: [2, 3, 4, 5],
  },
}

export default defineUserConfig({
  lang: 'zh-CN',
  title: '深度学习目标检测与跟踪',
  description: '使用文档和教程',
  base: "/deep-object-detect-track/",

  markdown: markdownOptions,
  theme: defaultTheme({
    // 在这里进行配置
    repo: 'https://github.com/HenryZhuHR/deep-object-detect-track',
    sidebarDepth: 3, // 设置根据页面标题自动生成的侧边栏的最大深度

  }),
})
