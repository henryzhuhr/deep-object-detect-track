import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/deep-object-detect-track/',
  title: "目标检测和跟踪",
  description: "目标检测和跟踪项目文档",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      { text: '项目源码', link: 'https://github.com/HenryZhuHR/deep-object-detect-track' },
      {
        text: '项目文档',
        items: [
          { text: '安装环境', link: '/install' },
          { text: '数据集制作', link: '/dataset' },
          { text: '模型训练', link: '/train' },
          { text: '模型部署', link: '/deploy' },
        ]
      }
    ],

    sidebar: [
      {
        text: '「项目文档」',
        items: [
          { text: '安装环境', link: '/install' },
          { text: '数据集制作', link: '/dataset' },
          { text: '模型训练', link: '/train' },
          { text: '模型部署', link: '/deploy' },
        ]
      }
    ],

    externalLinkIcon: true,
    footer: {
      message: '基于 <a href="https://choosealicense.com/licenses/gpl-3.0/">GPL-3.0</a> 许可发布',
      copyright: `版权所有 © 2024-${new Date().getFullYear()} <a href="https://github.com/HenryZhuHR?tab=repositories">HenryZhuHR</a>`
    },
    outline: {
      label: '页面导航'
    },
    
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },
    langMenuLabel: '多语言',
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '文档导航',
    darkModeSwitchLabel: '深色模式开关',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',

    socialLinks: [
      { icon: 'github', link: 'https://github.com/HenryZhuHR/deep-object-detect-track' }
    ]
  },
  lastUpdated: true,
  markdown: {
    math: true,
    lineNumbers: false
  }
})
