// pages/result/result.js

Page({
  data: {
    result: null,
    priorityClass: '',
    labelClass: '',
    riskClass: '',
    showFullReport: false,
    copied: false
  },

  onLoad(options) {
    if (options.data) {
      try {
        const result = JSON.parse(decodeURIComponent(options.data))
        this.setData({
          result: result,
          priorityClass: this.getPriorityClass(result.priority),
          labelClass: this.getLabelClass(result.cognitive_label),
          riskClass: this.getRiskClass(result.behavior_level)
        })
      } catch (e) {
        wx.showToast({ title: '数据解析失败', icon: 'none' })
      }
    }
  },

  getPriorityClass(priority) {
    const map = {
      '尽快劝阻': 'priority-urgent',
      '提醒留意': 'priority-notice',
      '告知误区': 'priority-inform',
      '无需处理': 'priority-safe'
    }
    return map[priority] || 'priority-safe'
  },

  getLabelClass(label) {
    const map = {
      '假': 'label-fake',
      '部分属实': 'label-partial',
      '无法核查': 'label-unknown',
      '真': 'label-true'
    }
    return map[label] || 'label-unknown'
  },

  getRiskClass(level) {
    const map = {
      '高': 'risk-high',
      '中': 'risk-mid',
      '低': 'risk-low'
    }
    return map[level] || 'risk-low'
  },

  copyRebuttal() {
    const text = this.data.result.rebuttal
    if (!text) return
    wx.setClipboardData({
      data: text,
      success: () => {
        this.setData({ copied: true })
        wx.showToast({ title: '已复制', icon: 'success' })
        setTimeout(() => this.setData({ copied: false }), 2000)
      }
    })
  },

  copyEvidence(e) {
    const url = e.currentTarget.dataset.url
    if (!url) return
    wx.setClipboardData({
      data: url,
      success: () => {
        wx.showToast({ title: '链接已复制', icon: 'success' })
      }
    })
  },

  toggleFullReport() {
    this.setData({ showFullReport: !this.data.showFullReport })
  },

  toggleRecognized() {
    this.setData({ showRecognized: !this.data.showRecognized })
  },

  goBack() {
    wx.navigateBack()
  },

  onShareAppMessage() {
    const r = this.data.result
    return {
      title: r ? `核查结果：${r.cognitive_label} · ${r.priority}` : '健康谣言核查结果',
      path: '/pages/index/index'
    }
  }
})
