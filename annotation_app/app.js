const ELEMENT_TYPES = ["TEXT", "IMAGE", "ICON", "SHAPE", "INPUT", "BUTTON_VISUAL"];
const GROUP_ROLES = [
  "CONTAINER", "CARD", "NAV", "FORM", "LIST", "LIST_ITEM",
  "TABLE", "TABLE_ROW", "SECTION", "UNKNOWN",
];
const LAYOUT_MODES = ["HORIZONTAL", "VERTICAL", "GRID", "FREE"];
const PRIMARY_ALIGNMENTS = ["START", "CENTER", "END", "SPACE_BETWEEN"];
const CROSS_ALIGNMENTS = ["START", "CENTER", "END", "STRETCH"];
const RESIZE_MODES = ["FIXED", "HUG", "STRETCH"];

const state = {
  assignment: null,
  mode: "adjudication",
  annotator: "annotator_a",
  sampleId: null,
  graph: null,
  annotation: null,
  selectedNodeIds: new Set(),
  selectedEntityIds: new Set(),
  activeEntityId: null,
  activeTokenId: null,
  scale: 1,
  dirty: false,
  validationErrors: [],
  nodeSearch: "",
  toastTimer: null,
  adjudication: null,
  references: { human: null, ai: null },
  differenceFilter: "all",
  activeDifferenceIndex: null,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function apiUrl(path) {
  return path.split("/").map((part, index) => (
    index === 0 ? part : encodeURIComponent(part)
  )).join("/");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({ error: "服务返回了无效 JSON" }));
  if (!response.ok && !Array.isArray(data.errors)) {
    throw new Error(data.error || `请求失败：${response.status}`);
  }
  return { response, data };
}

function toast(message, isError = false) {
  const element = $("toast");
  window.clearTimeout(state.toastTimer);
  element.textContent = message;
  element.classList.toggle("error", isError);
  element.classList.remove("hidden");
  state.toastTimer = window.setTimeout(() => element.classList.add("hidden"), 2800);
}

function getSample() {
  return state.assignment.samples.find((sample) => sample.sample_id === state.sampleId);
}

function isAdjudication() {
  return state.mode === "adjudication";
}

function isAdjudicationLocked() {
  return isAdjudication()
    && ["reviewed", "finalized"].includes(state.adjudication?.record?.status);
}

function isAnnotationLocked() {
  return !isAdjudication()
    && (state.assignment?.locked_sample_ids || []).includes(state.sampleId);
}

function isEditorLocked() {
  return isAdjudicationLocked() || isAnnotationLocked();
}

function entityById(id) {
  if (!state.annotation) return null;
  return state.annotation.elements.find((item) => item.id === id)
    || state.annotation.groups.find((item) => item.id === id)
    || null;
}

function tokenById(id) {
  return state.annotation?.style_tokens.find((item) => item.id === id) || null;
}

function nodeById(id) {
  return state.graph?.nodes.find((item) => item.id === Number(id)) || null;
}

function entityKind(id) {
  if (state.annotation.elements.some((item) => item.id === id)) return "element";
  if (state.annotation.groups.some((item) => item.id === id)) return "group";
  return null;
}

function uniqueId(prefix, items) {
  const ids = new Set(items.map((item) => item.id));
  let index = 1;
  while (ids.has(`${prefix}_${index}`)) index += 1;
  return `${prefix}_${index}`;
}

function bboxUnion(boxes) {
  if (!boxes.length) return { x: 0, y: 0, width: 0, height: 0 };
  const x1 = Math.min(...boxes.map((box) => box.x));
  const y1 = Math.min(...boxes.map((box) => box.y));
  const x2 = Math.max(...boxes.map((box) => box.x + box.width));
  const y2 = Math.max(...boxes.map((box) => box.y + box.height));
  return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
}

function updateGroupBBox(group) {
  const members = directChildIds(group.id).map(entityById).filter(Boolean);
  if (members.length) group.bbox = bboxUnion(members.map((item) => item.bbox));
}

function currentParent(id) {
  return state.annotation.tree.find((edge) => edge.child_id === id)?.parent_id || "page_root";
}

function descendantsOf(id) {
  const result = new Set();
  const visit = (parentId) => {
    state.annotation.tree
      .filter((edge) => edge.parent_id === parentId)
      .forEach((edge) => {
        if (!result.has(edge.child_id)) {
          result.add(edge.child_id);
          visit(edge.child_id);
        }
      });
  };
  visit(id);
  return result;
}

function directChildIds(parentId) {
  return state.annotation.tree
    .filter((edge) => edge.parent_id === parentId)
    .sort((left, right) => (left.order - right.order) || left.child_id.localeCompare(right.child_id))
    .map((edge) => edge.child_id);
}

function descendantElementIds(entityId, visiting = new Set()) {
  if (entityKind(entityId) === "element") return [entityId];
  if (entityKind(entityId) !== "group" || visiting.has(entityId)) return [];
  const nextVisiting = new Set(visiting);
  nextVisiting.add(entityId);
  return [...new Set(
    directChildIds(entityId)
      .flatMap((childId) => descendantElementIds(childId, nextVisiting)),
  )];
}

function refreshGroupSourceElements() {
  state.annotation.groups.forEach((group) => {
    group.source_element_ids = descendantElementIds(group.id);
  });
}

function setParent(childId, parentId) {
  const descendants = descendantsOf(childId);
  if (parentId === childId || descendants.has(parentId)) {
    toast("不能把实体放入自身或其后代分组", true);
    return false;
  }
  state.annotation.tree = state.annotation.tree.filter((edge) => edge.child_id !== childId);
  state.annotation.tree.push({
    parent_id: parentId || "page_root",
    child_id: childId,
    order: 0,
    confidence: 1,
  });
  normalizeTree();
  refreshGroupSourceElements();
  markDirty();
  return true;
}

function normalizeTree() {
  const validIds = new Set([
    ...state.annotation.elements.map((item) => item.id),
    ...state.annotation.groups.map((item) => item.id),
  ]);
  const groupIds = new Set(state.annotation.groups.map((item) => item.id));
  const seenChildren = new Set();
  state.annotation.tree = state.annotation.tree.filter((edge) => {
    const valid = validIds.has(edge.child_id)
      && (edge.parent_id === "page_root" || groupIds.has(edge.parent_id))
      && edge.child_id !== edge.parent_id
      && !seenChildren.has(edge.child_id);
    if (valid) seenChildren.add(edge.child_id);
    return valid;
  });
  validIds.forEach((id) => {
    if (!seenChildren.has(id)) {
      state.annotation.tree.push({
        parent_id: "page_root", child_id: id, order: 0, confidence: 1,
      });
    }
  });

  const byParent = new Map();
  state.annotation.tree.forEach((edge) => {
    if (!byParent.has(edge.parent_id)) byParent.set(edge.parent_id, []);
    byParent.get(edge.parent_id).push(edge);
  });
  byParent.forEach((edges) => {
    edges.sort((left, right) => {
      const a = entityById(left.child_id)?.bbox || { x: 0, y: 0 };
      const b = entityById(right.child_id)?.bbox || { x: 0, y: 0 };
      return (a.y - b.y) || (a.x - b.x) || left.child_id.localeCompare(right.child_id);
    });
    edges.forEach((edge, index) => { edge.order = index; });
  });
}

function normalizeAnnotation() {
  const elementIds = new Set(state.annotation.elements.map((item) => item.id));
  const groupIds = new Set(state.annotation.groups.map((item) => item.id));
  const entityIds = new Set([...elementIds, ...groupIds]);

  state.annotation.layouts = state.annotation.layouts
    .filter((layout) => groupIds.has(layout.target_id))
    .filter((layout, index, all) => (
      all.findIndex((item) => item.target_id === layout.target_id) === index
    ));
  state.annotation.groups.forEach((group) => {
    if (!state.annotation.layouts.some((layout) => layout.target_id === group.id)) {
      state.annotation.layouts.push(defaultLayout(group.id));
    }
  });
  state.annotation.style_tokens.forEach((token) => {
    token.member_ids = [...new Set(token.member_ids)].filter((id) => entityIds.has(id));
  });
  normalizeTree();
  refreshGroupSourceElements();
}

function markDirty() {
  if (isEditorLocked()) {
    toast(isAdjudication()
      ? "该样本已完成审核，不能继续修改"
      : "该样本已冻结为金标准，不能继续修改", true);
    return;
  }
  state.dirty = true;
  renderSaveState();
}

function renderSaveState() {
  const badge = $("saveState");
  const complete = state.annotation?.provenance?.status === "complete";
  badge.classList.toggle("dirty", state.dirty);
  badge.classList.toggle("complete", !state.dirty && complete);
  badge.textContent = state.dirty ? "未保存" : (complete ? "已完成" : "已保存");
}

function renderProgress() {
  const progress = isAdjudication()
    ? state.assignment.adjudication.progress
    : state.assignment.progress[state.annotator];
  $("progressText").textContent = isAdjudication()
    ? `${progress.reviewed} / ${progress.total}`
    : `${progress.complete} / ${progress.total}`;
  [...$("sampleSelect").options].forEach((option) => {
    const status = isAdjudication()
      ? state.assignment.adjudication.sample_status[option.value]
      : state.assignment.sample_status[state.annotator][option.value];
    const sample = state.assignment.samples.find((item) => item.sample_id === option.value);
    const complete = isAdjudication()
      ? ["reviewed", "finalized"].includes(status)
      : status === "complete";
    option.textContent = `${complete ? "✓ " : ""}${option.value} · ${sample.size_bin}`;
  });
}

function renderValidation() {
  $("validationCount").textContent = String(state.validationErrors.length);
  if (!state.validationErrors.length) {
    $("validationList").innerHTML = '<div class="validation-ok">当前没有校验错误</div>';
    return;
  }
  $("validationList").innerHTML = state.validationErrors
    .map((error) => `<div class="validation-error">${escapeHtml(error)}</div>`)
    .join("");
}

function usedNodeIds() {
  return new Set(state.annotation.elements.flatMap((item) => item.source_node_ids));
}

function toggleNode(nodeId) {
  if (usedNodeIds().has(nodeId) && !state.selectedNodeIds.has(nodeId)) {
    toast("该源节点已属于其他原子元素", true);
    return;
  }
  if (state.selectedNodeIds.has(nodeId)) state.selectedNodeIds.delete(nodeId);
  else state.selectedNodeIds.add(nodeId);
  if (state.selectedNodeIds.size === 1) {
    $("newElementType").value = inferElementType(nodeById(nodeId));
  }
  renderNodes();
  renderCanvas();
}

function nodeLabel(node) {
  const semantic = node.attributes?.["aria-label"]
    || node.attributes?.alt
    || node.attributes?.title
    || node.attributes?.name
    || node.attributes?.id
    || node.text;
  return semantic?.trim().slice(0, 44) || `${node.tag} ${node.id}`;
}

function inferElementType(node) {
  if (!node) return "SHAPE";
  if (["img", "picture", "video", "canvas"].includes(node.tag)) return "IMAGE";
  if (["svg", "i"].includes(node.tag)) return "ICON";
  if (["input", "textarea", "select"].includes(node.tag)) return "INPUT";
  if (node.tag === "button" || node.attributes?.role === "button") return "BUTTON_VISUAL";
  if (["p", "span", "label", "a", "h1", "h2", "h3", "h4", "h5", "h6", "li"].includes(node.tag)
      && node.text?.trim()) return "TEXT";
  return "SHAPE";
}

function styleFromNode(node) {
  const style = node?.computed_style || {};
  const result = {};
  const mapping = {
    backgroundColor: "background_color",
    color: "text_color",
    borderColor: "border_color",
    borderRadius: "border_radius",
    fontFamily: "font_family",
    fontSize: "font_size",
    fontWeight: "font_weight",
    lineHeight: "line_height",
    opacity: "opacity",
  };
  Object.entries(mapping).forEach(([source, target]) => {
    if (style[source] != null && style[source] !== "") result[target] = style[source];
  });
  const imageSrc = node?.attributes?.src || node?.attributes?.["data-src"];
  if (imageSrc) result.image_src = imageSrc;
  return result;
}

function renderNodes() {
  if (!state.graph) return;
  const query = state.nodeSearch.trim().toLowerCase();
  const used = usedNodeIds();
  const nodes = state.graph.nodes.filter((node) => {
    if (!query) return true;
    return [
      node.tag, node.text, node.attributes?.id, node.attributes?.class,
      node.attributes?.["aria-label"],
    ].some((value) => String(value || "").toLowerCase().includes(query));
  });
  $("nodeSelectionCount").textContent = String(state.selectedNodeIds.size);
  $("nodeList").innerHTML = nodes.map((node) => `
    <button class="list-row ${state.selectedNodeIds.has(node.id) ? "active" : ""}"
      type="button" data-node-id="${node.id}">
      <input type="checkbox" tabindex="-1" ${state.selectedNodeIds.has(node.id) ? "checked" : ""}
        ${used.has(node.id) && !state.selectedNodeIds.has(node.id) ? "disabled" : ""}>
      <span class="row-copy">
        <span class="row-title">${escapeHtml(nodeLabel(node))}</span>
        <span class="row-subtitle">#${node.id} · ${escapeHtml(node.tag)} · ${Math.round(node.bbox.width)}×${Math.round(node.bbox.height)}</span>
      </span>
      <span class="row-badge">${used.has(node.id) ? "USED" : `D${node.depth}`}</span>
    </button>
  `).join("") || '<div class="empty-state compact">没有匹配节点</div>';
  $("nodeList").querySelectorAll("[data-node-id]").forEach((row) => {
    row.addEventListener("click", () => toggleNode(Number(row.dataset.nodeId)));
  });
}

function referenceEntities(annotation, source) {
  if (!annotation) return [];
  return [
    ...annotation.groups.map((entity) => ({ entity, kind: "group", source })),
    ...annotation.elements.map((entity) => ({ entity, kind: "element", source })),
  ];
}

function renderReferenceOverlay() {
  if (!isAdjudication()) {
    $("referenceOverlay").innerHTML = "";
    return;
  }
  const entities = [];
  if ($("showHumanReference").checked) {
    entities.push(...referenceEntities(state.references.human, "human"));
  }
  if ($("showAiReference").checked) {
    entities.push(...referenceEntities(state.references.ai, "ai"));
  }
  const focusedSources = new Set(state.selectedNodeIds);
  $("referenceOverlay").innerHTML = entities.map(({ entity, kind, source }, index) => {
    const box = entity.bbox;
    const sourceIds = entitySourceIds(
      source === "human" ? state.references.human : state.references.ai,
      entity,
    );
    const focused = sourceIds.some((id) => focusedSources.has(id)) ? "focused" : "";
    return `<div class="reference-box ${source} ${kind} ${focused}"
      title="${source === "human" ? "人工" : "AI"} · ${escapeHtml(entity.name)}"
      style="left:${box.x * state.scale}px;top:${box.y * state.scale}px;width:${Math.max(2, box.width * state.scale)}px;height:${Math.max(2, box.height * state.scale)}px;z-index:${50 + index}"></div>`;
  }).join("");
}

function renderCanvas() {
  if (!state.graph) return;
  const { width, height } = state.graph.canvas;
  const scaledWidth = Math.max(1, width * state.scale);
  const scaledHeight = Math.max(1, height * state.scale);
  const stage = $("canvasStage");
  stage.style.width = `${scaledWidth}px`;
  stage.style.height = `${scaledHeight}px`;
  $("zoomValue").textContent = `${Math.round(state.scale * 100)}%`;

  const used = usedNodeIds();
  const showAll = $("showAllNodes").checked;
  const showLabels = $("showLabels").checked;
  const nodes = [...state.graph.nodes].sort((a, b) => (
    (b.bbox.width * b.bbox.height) - (a.bbox.width * a.bbox.height)
  ));
  $("nodeOverlay").innerHTML = nodes
    .filter((node) => showAll || state.selectedNodeIds.has(node.id))
    .map((node, index) => {
      const box = node.bbox;
      const classes = [
        "node-box",
        state.selectedNodeIds.has(node.id) ? "selected" : "",
        used.has(node.id) ? "used" : "",
      ].filter(Boolean).join(" ");
      return `<button type="button" class="${classes}" data-canvas-node-id="${node.id}"
        title="#${node.id} ${escapeHtml(node.tag)} ${escapeHtml(nodeLabel(node))}"
        style="left:${box.x * state.scale}px;top:${box.y * state.scale}px;width:${Math.max(2, box.width * state.scale)}px;height:${Math.max(2, box.height * state.scale)}px;z-index:${index + 1}">
        ${showLabels ? `<span class="node-label">#${node.id} ${escapeHtml(node.tag)}</span>` : ""}
      </button>`;
    }).join("");
  $("nodeOverlay").querySelectorAll("[data-canvas-node-id]").forEach((box) => {
    box.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleNode(Number(box.dataset.canvasNodeId));
    });
  });

  renderReferenceOverlay();

  const entities = [
    ...state.annotation.groups.map((entity) => ({ entity, kind: "group" })),
    ...state.annotation.elements.map((entity) => ({ entity, kind: "element" })),
  ];
  $("entityOverlay").innerHTML = entities.map(({ entity, kind }, index) => {
    const box = entity.bbox;
    const active = entity.id === state.activeEntityId ? "active" : "";
    return `<div class="entity-box ${kind} ${active}"
      title="${escapeHtml(entity.name)}"
      style="left:${box.x * state.scale}px;top:${box.y * state.scale}px;width:${Math.max(2, box.width * state.scale)}px;height:${Math.max(2, box.height * state.scale)}px;z-index:${100 + index}"></div>`;
  }).join("");
}

function createElement() {
  const nodes = [...state.selectedNodeIds].map(nodeById).filter(Boolean);
  if (!nodes.length) {
    toast("请先选择至少一个实现节点", true);
    return;
  }
  const selectedIds = nodes.map((node) => node.id);
  const taken = usedNodeIds();
  if (selectedIds.some((id) => taken.has(id))) {
    toast("选择中包含已使用节点", true);
    return;
  }
  const first = nodes[0];
  const id = uniqueId("e", state.annotation.elements);
  const texts = [...new Set(nodes.map((node) => node.text?.trim()).filter(Boolean))];
  state.annotation.elements.push({
    id,
    source_node_ids: selectedIds,
    type: $("newElementType").value,
    bbox: bboxUnion(nodes.map((node) => node.bbox)),
    name: nodeLabel(first),
    text: texts.join(" ").slice(0, 500),
    style: styleFromNode(first),
    style_token_refs: [],
    confidence: 1,
  });
  state.annotation.tree.push({
    parent_id: "page_root", child_id: id, order: 0, confidence: 1,
  });
  state.selectedNodeIds.clear();
  state.activeEntityId = id;
  state.activeTokenId = null;
  markDirty();
  renderAll();
  toast(`已创建元素 ${id}`);
}

function renderEntities() {
  $("entitySelectionCount").textContent = String(state.selectedEntityIds.size);
  const rows = [
    ...state.annotation.groups.map((item) => ({ item, badge: item.role, group: true })),
    ...state.annotation.elements.map((item) => ({ item, badge: item.type, group: false })),
  ];
  $("entityList").innerHTML = rows.map(({ item, badge, group }) => `
    <button class="list-row ${state.activeEntityId === item.id ? "active" : ""}"
      type="button" data-entity-id="${escapeHtml(item.id)}">
      <input class="entity-checkbox" type="checkbox" tabindex="-1"
        ${state.selectedEntityIds.has(item.id) ? "checked" : ""}>
      <span class="row-copy">
        <span class="row-title">${escapeHtml(item.name || item.id)}</span>
        <span class="row-subtitle">${escapeHtml(item.id)} · ${group ? `${directChildIds(item.id).length} 直接子项` : `${item.source_node_ids.length} 源节点`}</span>
      </span>
      <span class="row-badge">${escapeHtml(badge)}</span>
    </button>
  `).join("") || '<div class="empty-state compact">尚未创建实体</div>';
  $("entityList").querySelectorAll("[data-entity-id]").forEach((row) => {
    row.addEventListener("click", (event) => {
      const id = row.dataset.entityId;
      if (event.target.classList.contains("entity-checkbox")) {
        if (state.selectedEntityIds.has(id)) state.selectedEntityIds.delete(id);
        else state.selectedEntityIds.add(id);
        renderEntities();
        renderTokens();
        return;
      }
      state.activeEntityId = id;
      state.activeTokenId = null;
      renderEntities();
      renderCanvas();
      renderInspector();
    });
  });
}

function defaultLayout(targetId) {
  return {
    target_id: targetId,
    mode: "FREE",
    gap: 0,
    padding: [0, 0, 0, 0],
    primary_align: "START",
    cross_align: "START",
    horizontal_resize: "FIXED",
    vertical_resize: "FIXED",
    confidence: 1,
  };
}

function createGroup() {
  const memberIds = [...state.selectedEntityIds]
    .filter((id) => entityKind(id));
  if (!memberIds.length) {
    toast("请先选择设计元素或子分组", true);
    return;
  }
  const containsAncestorPair = memberIds.some((left) => (
    memberIds.some((right) => left !== right && descendantsOf(left).has(right))
  ));
  if (containsAncestorPair) {
    toast("不能同时选择祖先分组及其后代实体", true);
    return;
  }
  const members = memberIds.map(entityById);
  const id = uniqueId("g", state.annotation.groups);
  state.annotation.groups.push({
    id,
    source_element_ids: [],
    role: "CONTAINER",
    bbox: bboxUnion(members.map((item) => item.bbox)),
    name: `group ${state.annotation.groups.length + 1}`,
    style: {},
    source_node_id: null,
    confidence: 1,
  });
  state.annotation.layouts.push(defaultLayout(id));
  const memberParents = memberIds.map(currentParent);
  const sharedParent = memberParents.every((value) => value === memberParents[0])
    ? memberParents[0] : "page_root";
  state.annotation.tree.push({
    parent_id: sharedParent, child_id: id, order: 0, confidence: 1,
  });
  memberIds.forEach((memberId) => setParent(memberId, id));
  normalizeAnnotation();
  state.selectedEntityIds.clear();
  state.activeEntityId = id;
  state.activeTokenId = null;
  markDirty();
  renderAll();
  toast(`已创建分组 ${id}`);
}

function parentOptions(entityId) {
  const invalid = descendantsOf(entityId);
  return [
    { value: "page_root", label: "画布（page_root）" },
    ...state.annotation.groups
      .filter((group) => group.id !== entityId && !invalid.has(group.id))
      .map((group) => ({ value: group.id, label: `${group.name} · ${group.id}` })),
  ];
}

function optionsHtml(values, selected) {
  return values.map((value) => {
    const option = typeof value === "string" ? { value, label: value } : value;
    return `<option value="${escapeHtml(option.value)}" ${option.value === selected ? "selected" : ""}>${escapeHtml(option.label)}</option>`;
  }).join("");
}

function numericField(label, path, value, step = "1") {
  return `<label><span>${label}</span><input type="number" step="${step}" data-path="${path}" value="${Number(value)}"></label>`;
}

function renderInspector() {
  const entity = entityById(state.activeEntityId);
  const token = tokenById(state.activeTokenId);
  $("inspectorEmpty").classList.toggle("hidden", Boolean(entity || token));
  $("entityInspector").classList.toggle("hidden", !entity);
  $("tokenInspector").classList.toggle("hidden", !token);
  $("deleteEntity").classList.toggle("hidden", !entity && !token);

  if (entity) {
    const kind = entityKind(entity.id);
    $("inspectorTitle").textContent = `${entity.name || entity.id} · ${entity.id}`;
    $("deleteEntity").textContent = kind === "group" ? "删除分组" : "删除元素";
    renderEntityInspector(entity, kind);
  } else if (token) {
    $("inspectorTitle").textContent = `${token.name} · ${token.id}`;
    $("deleteEntity").textContent = "删除 Token";
    renderTokenInspector(token);
  } else {
    $("inspectorTitle").textContent = "未选择实体";
  }
}

function renderEntityInspector(entity, kind) {
  const isGroup = kind === "group";
  const typeValues = isGroup ? GROUP_ROLES : ELEMENT_TYPES;
  const typeValue = isGroup ? entity.role : entity.type;
  let html = `
    <div class="field-grid">
      <label class="wide"><span>名称</span><input data-path="name" value="${escapeHtml(entity.name)}"></label>
      <label><span>${isGroup ? "语义角色" : "元素类型"}</span>
        <select data-path="${isGroup ? "role" : "type"}">${optionsHtml(typeValues, typeValue)}</select>
      </label>
      <label><span>父级</span>
        <select data-parent>${optionsHtml(parentOptions(entity.id), currentParent(entity.id))}</select>
      </label>
    </div>`;
  if (!isGroup) {
    html += `
      <label class="inspector-wide"><span>文本</span><textarea data-path="text">${escapeHtml(entity.text)}</textarea></label>
      <div class="read-only-line">源节点：${entity.source_node_ids.map((id) => `#${id}`).join(", ")}</div>`;
  }
  html += `
    <div class="section-label">边界框</div>
    <div class="field-grid">
      ${numericField("X", "bbox.x", entity.bbox.x, "0.1")}
      ${numericField("Y", "bbox.y", entity.bbox.y, "0.1")}
      ${numericField("宽", "bbox.width", entity.bbox.width, "0.1")}
      ${numericField("高", "bbox.height", entity.bbox.height, "0.1")}
    </div>`;
  if (isGroup) {
    const layout = state.annotation.layouts.find((item) => item.target_id === entity.id)
      || defaultLayout(entity.id);
    const children = directChildIds(entity.id);
    const childCandidates = [
      ...state.annotation.groups
        .filter((group) => group.id !== entity.id),
      ...state.annotation.elements,
    ];
    html += `
      <div class="section-label">直接子实体</div>
      <div class="member-list">${childCandidates.map((child) => `
        <label class="member-option">
          <input type="checkbox" data-group-child="${escapeHtml(child.id)}"
            ${children.includes(child.id) ? "checked" : ""}>
          <span>${escapeHtml(child.name)} · ${escapeHtml(child.id)} · ${entityKind(child.id) === "group" ? "GROUP" : child.type}</span>
        </label>`).join("")}</div>
      <div class="read-only-line">后代原子覆盖（自动）：${entity.source_element_ids.length ? entity.source_element_ids.join(", ") : "无"}</div>
      <div class="section-label">布局约束</div>
      <div class="field-grid">
        <label><span>方向</span><select data-layout="mode">${optionsHtml(LAYOUT_MODES, layout.mode)}</select></label>
        ${numericField("间距", "layout.gap", layout.gap, "0.1")}
        <label><span>主轴对齐</span><select data-layout="primary_align">${optionsHtml(PRIMARY_ALIGNMENTS, layout.primary_align)}</select></label>
        <label><span>交叉轴对齐</span><select data-layout="cross_align">${optionsHtml(CROSS_ALIGNMENTS, layout.cross_align)}</select></label>
        <label><span>水平尺寸</span><select data-layout="horizontal_resize">${optionsHtml(RESIZE_MODES, layout.horizontal_resize)}</select></label>
        <label><span>垂直尺寸</span><select data-layout="vertical_resize">${optionsHtml(RESIZE_MODES, layout.vertical_resize)}</select></label>
      </div>
      <div class="section-label">内边距（上 / 右 / 下 / 左）</div>
      <div class="quad-grid">
        ${[0, 1, 2, 3].map((index) => `<input type="number" step="0.1" data-padding="${index}" value="${Number(layout.padding[index] || 0)}">`).join("")}
      </div>`;
  }
  $("entityInspector").innerHTML = html;
  bindEntityInspector(entity, isGroup);
}

function assignPath(object, path, value) {
  const parts = path.split(".");
  const final = parts.pop();
  const target = parts.reduce((current, part) => current[part], object);
  target[final] = value;
}

function inputValue(input) {
  return input.type === "number" ? Number(input.value) : input.value;
}

function bindEntityInspector(entity, isGroup) {
  const form = $("entityInspector");
  form.querySelectorAll("[data-path]").forEach((input) => {
    if (input.dataset.path.startsWith("layout.")) return;
    input.addEventListener("change", () => {
      assignPath(entity, input.dataset.path, inputValue(input));
      markDirty();
      renderEntities();
      renderCanvas();
      renderInspector();
    });
  });
  form.querySelector("[data-parent]")?.addEventListener("change", (event) => {
    if (setParent(entity.id, event.target.value)) {
      renderEntities();
      renderInspector();
    }
  });
  if (!isGroup) return;
  const layout = state.annotation.layouts.find((item) => item.target_id === entity.id);
  form.querySelectorAll("[data-group-child]").forEach((input) => {
    input.addEventListener("change", () => {
      const memberId = input.dataset.groupChild;
      const groupParent = currentParent(entity.id);
      if (input.checked) {
        if (!setParent(memberId, entity.id)) {
          renderInspector();
          return;
        }
      } else {
        if (currentParent(memberId) === entity.id) {
          setParent(memberId, groupParent);
        }
      }
      normalizeAnnotation();
      updateGroupBBox(entity);
      markDirty();
      renderAll();
    });
  });
  form.querySelectorAll("[data-layout]").forEach((input) => {
    input.addEventListener("change", () => {
      layout[input.dataset.layout] = inputValue(input);
      markDirty();
      renderInspector();
    });
  });
  form.querySelectorAll("[data-path='layout.gap']").forEach((input) => {
    input.addEventListener("change", () => {
      layout.gap = Number(input.value);
      markDirty();
      renderInspector();
    });
  });
  form.querySelectorAll("[data-padding]").forEach((input) => {
    input.addEventListener("change", () => {
      layout.padding[Number(input.dataset.padding)] = Number(input.value);
      markDirty();
      renderInspector();
    });
  });
}

function rgbaToHex(color) {
  if (!Array.isArray(color) || color.length < 3) return "#2563eb";
  return `#${color.slice(0, 3).map((value) => (
    Math.round(clamp(Number(value), 0, 1) * 255).toString(16).padStart(2, "0")
  )).join("")}`;
}

function defaultTokenValue(kind) {
  if (kind === "COLOR") return { property: "background", rgba: [0.145, 0.388, 0.922, 1] };
  if (kind === "TEXT") return { font_size: 16, font_weight: 400 };
  if (kind === "RADIUS") return { radius: 4 };
  return { spacing: 8 };
}

function roundedNumber(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}

function normalizedColor(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  const rgba = value.slice(0, 4).map((channel) => roundedNumber(channel, 3));
  if (rgba.length === 3) rgba.push(1);
  return rgba[3] <= 0.05 ? null : rgba;
}

function candidateEntries(entity) {
  const result = [];
  const style = entity.style || {};
  const background = normalizedColor(style.background_color);
  const foreground = normalizedColor(style.text_color);
  const border = normalizedColor(style.border_color);
  if (background) result.push({
    kind: "COLOR", value: { property: "background", rgba: background }, label: "背景色",
  });
  if (foreground && (entityKind(entity.id) === "group" || entity.text || entity.type === "TEXT")) {
    result.push({
      kind: "COLOR", value: { property: "foreground", rgba: foreground }, label: "文字色",
    });
  }
  if (border && roundedNumber(style.border_width) > 0) result.push({
    kind: "COLOR", value: { property: "border", rgba: border }, label: "边框色",
  });
  if (style.font_size != null && (entity.text || entity.type === "TEXT")) result.push({
    kind: "TEXT",
    value: {
      font_size: roundedNumber(style.font_size),
      font_weight: Math.round(Number(style.font_weight || 400)),
    },
    label: "文字样式",
  });
  if (roundedNumber(style.border_radius) > 0) result.push({
    kind: "RADIUS", value: { radius: roundedNumber(style.border_radius) }, label: "圆角",
  });
  if (entityKind(entity.id) === "group") {
    const layout = layoutByTarget(entity.id);
    if (roundedNumber(layout?.gap) > 0) result.push({
      kind: "SPACING",
      value: { property: "gap", spacing: roundedNumber(layout.gap) },
      label: "项目间距",
    });
    const padding = layout?.padding || [];
    if (padding.length === 4 && padding.every((item) => roundedNumber(item) === roundedNumber(padding[0]))
      && roundedNumber(padding[0]) > 0) result.push({
      kind: "SPACING",
      value: { property: "padding", spacing: roundedNumber(padding[0]) },
      label: "内边距",
    });
  }
  return result;
}

function tokenCandidateKey(candidate) {
  return `${candidate.kind}|${JSON.stringify(candidate.value)}|${[...candidate.member_ids].sort().join(",")}`;
}

function tokenCandidates() {
  const buckets = new Map();
  [...state.annotation.elements, ...state.annotation.groups].forEach((entity) => {
    candidateEntries(entity).forEach((entry) => {
      const valueKey = `${entry.kind}|${JSON.stringify(entry.value)}`;
      if (!buckets.has(valueKey)) buckets.set(valueKey, { ...entry, member_ids: [] });
      buckets.get(valueKey).member_ids.push(entity.id);
    });
  });
  return [...buckets.values()]
    .map((candidate) => ({
      ...candidate,
      member_ids: [...new Set(candidate.member_ids)].sort(),
    }))
    .filter((candidate) => candidate.member_ids.length >= 2)
    .map((candidate) => ({ ...candidate, key: tokenCandidateKey(candidate) }))
    .sort((left, right) => left.kind.localeCompare(right.kind)
      || right.member_ids.length - left.member_ids.length
      || left.key.localeCompare(right.key));
}

function tokenReview() {
  const provenance = state.annotation.provenance;
  if (!provenance.token_review || typeof provenance.token_review !== "object") {
    provenance.token_review = {
      status: "pending", accepted_candidate_keys: [], ignored_candidate_keys: [],
    };
  }
  const review = provenance.token_review;
  review.accepted_candidate_keys = [...new Set(review.accepted_candidate_keys || [])];
  review.ignored_candidate_keys = [...new Set(review.ignored_candidate_keys || [])];
  return review;
}

function tokenValueLabel(candidate) {
  if (candidate.kind === "COLOR") return rgbaToHex(candidate.value.rgba);
  if (candidate.kind === "TEXT") {
    return `${candidate.value.font_size}px / ${candidate.value.font_weight}`;
  }
  if (candidate.kind === "RADIUS") return `${candidate.value.radius}px`;
  return `${candidate.value.spacing}px`;
}

function automaticTokenName(candidate) {
  const index = state.annotation.style_tokens.filter((token) => token.kind === candidate.kind).length + 1;
  return `${candidate.label} ${index}`;
}

function addToken(candidate, candidateKey = null) {
  const id = uniqueId(`token_${candidate.kind.toLowerCase()}`, state.annotation.style_tokens);
  state.annotation.style_tokens.push({
    id,
    kind: candidate.kind,
    value: structuredClone(candidate.value),
    member_ids: [...candidate.member_ids],
    name: automaticTokenName(candidate),
    confidence: 1,
  });
  if (candidateKey) {
    const review = tokenReview();
    review.accepted_candidate_keys.push(candidateKey);
    review.ignored_candidate_keys = review.ignored_candidate_keys.filter((key) => key !== candidateKey);
  }
  state.activeTokenId = id;
  state.activeEntityId = null;
  return id;
}

function acceptTokenCandidate(key) {
  const candidate = tokenCandidates().find((item) => item.key === key);
  if (!candidate) return;
  addToken(candidate, key);
  markDirty();
  renderAll();
}

function ignoreTokenCandidate(key) {
  const review = tokenReview();
  review.ignored_candidate_keys.push(key);
  review.accepted_candidate_keys = review.accepted_candidate_keys.filter((item) => item !== key);
  const removedToken = state.annotation.style_tokens.find(
    (token) => tokenCandidateKey(token) === key,
  );
  state.annotation.style_tokens = state.annotation.style_tokens.filter(
    (token) => tokenCandidateKey(token) !== key,
  );
  if (removedToken?.id === state.activeTokenId) state.activeTokenId = null;
  markDirty();
  renderAll();
}

function completeTokenReview() {
  const review = tokenReview();
  const decided = new Set([
    ...review.accepted_candidate_keys, ...review.ignored_candidate_keys,
  ]);
  tokenCandidates().forEach((candidate) => {
    if (!decided.has(candidate.key)) review.ignored_candidate_keys.push(candidate.key);
  });
  review.status = "reviewed";
  review.reviewed_candidate_keys = tokenCandidates().map((candidate) => candidate.key);
  review.reviewed_at = new Date().toISOString();
  markDirty();
  renderTokens();
  toast(state.annotation.style_tokens.length
    ? `Token 检查完成，保留 ${state.annotation.style_tokens.length} 个`
    : "Token 检查完成：本页无共享 Token");
}

function createToken() {
  const memberIds = [...state.selectedEntityIds].filter((id) => entityById(id));
  if (memberIds.length < 2) {
    toast("创建 Token 至少需要选择两个实体", true);
    return;
  }
  const kind = $("tokenKind").value;
  const entries = candidateEntries(entityById(memberIds[0]));
  const inferred = entries.find((entry) => entry.kind === kind);
  const candidate = {
    kind,
    value: inferred?.value || defaultTokenValue(kind),
    member_ids: memberIds,
    label: inferred?.label || ({
      COLOR: "颜色", TEXT: "文字样式", RADIUS: "圆角", SPACING: "间距",
    })[kind],
  };
  const id = addToken(candidate);
  state.selectedEntityIds.clear();
  markDirty();
  renderAll();
  toast(`已创建 Token ${id}`);
}

function renderTokens() {
  const review = tokenReview();
  const candidates = tokenCandidates();
  const reviewedKeys = [...(review.reviewed_candidate_keys || [])].sort();
  const currentKeys = candidates.map((candidate) => candidate.key).sort();
  const reviewIsCurrent = review.status === "reviewed"
    && JSON.stringify(reviewedKeys) === JSON.stringify(currentKeys);
  const accepted = new Set(review.accepted_candidate_keys);
  const ignored = new Set(review.ignored_candidate_keys);
  $("tokenReviewStatus").textContent = reviewIsCurrent ? "已检查" : (
    review.status === "reviewed" ? "需复查" : "待检查"
  );
  $("tokenReviewStatus").classList.toggle("complete", reviewIsCurrent);
  $("completeTokenReview").disabled = isEditorLocked();
  $("tokenCandidateList").innerHTML = candidates.map((candidate) => {
    const status = accepted.has(candidate.key) ? "accepted" : (ignored.has(candidate.key) ? "ignored" : "pending");
    return `<div class="token-candidate ${status}">
      <div class="token-candidate-main">
        ${candidate.kind === "COLOR" ? `<span class="token-swatch" style="background:${rgbaToHex(candidate.value.rgba)}"></span>` : ""}
        <span class="row-copy">
          <span class="row-title">${escapeHtml(candidate.label)} · ${escapeHtml(tokenValueLabel(candidate))}</span>
          <span class="row-subtitle">${candidate.member_ids.length} 个成员 · ${escapeHtml(candidate.member_ids.join("、"))}</span>
        </span>
        <span class="row-badge">${escapeHtml(candidate.kind)}</span>
      </div>
      <div class="token-candidate-actions">
        <span class="candidate-decision">${status === "accepted" ? "已保留" : (status === "ignored" ? "已忽略" : "未决定")}</span>
        ${status !== "ignored" ? `<button type="button" data-ignore-token-candidate="${escapeHtml(candidate.key)}">${status === "accepted" ? "改为忽略" : "忽略"}</button>` : ""}
        ${status !== "accepted" ? `<button class="primary-button" type="button" data-accept-token-candidate="${escapeHtml(candidate.key)}">${status === "ignored" ? "改为保留" : "保留"}</button>` : ""}
      </div>
    </div>`;
  }).join("") || '<div class="empty-state compact">没有重复样式候选</div>';
  $("tokenCandidateList").querySelectorAll("[data-accept-token-candidate]").forEach((button) => {
    button.addEventListener("click", () => acceptTokenCandidate(button.dataset.acceptTokenCandidate));
  });
  $("tokenCandidateList").querySelectorAll("[data-ignore-token-candidate]").forEach((button) => {
    button.addEventListener("click", () => ignoreTokenCandidate(button.dataset.ignoreTokenCandidate));
  });
  $("tokenList").innerHTML = state.annotation.style_tokens.map((token) => `
    <button type="button" class="token-row ${state.activeTokenId === token.id ? "active" : ""}"
      data-token-id="${escapeHtml(token.id)}">
      <span class="token-row-main">
        ${token.kind === "COLOR" ? `<span class="token-swatch" style="background:${rgbaToHex(token.value.rgba)}"></span>` : ""}
        <span class="row-copy">
          <span class="row-title">${escapeHtml(token.name)}</span>
          <span class="row-subtitle">${escapeHtml(token.id)} · ${token.member_ids.length} 成员</span>
        </span>
      </span>
      <span class="row-badge">${escapeHtml(token.kind)}</span>
    </button>
  `).join("") || '<div class="empty-state compact">尚未创建 Token</div>';
  $("tokenList").querySelectorAll("[data-token-id]").forEach((row) => {
    row.addEventListener("click", () => {
      state.activeTokenId = row.dataset.tokenId;
      state.activeEntityId = null;
      renderTokens();
      renderEntities();
      renderCanvas();
      renderInspector();
    });
  });
}

function renderTokenInspector(token) {
  const allEntities = [...state.annotation.elements, ...state.annotation.groups];
  $("tokenInspector").innerHTML = `
    <div class="field-grid">
      <label class="wide"><span>名称</span><input data-token-name value="${escapeHtml(token.name)}"></label>
      <label><span>类型</span><input value="${escapeHtml(token.kind)}" disabled></label>
      <label><span>自动提取值</span><input value="${escapeHtml(tokenValueLabel(token))}" disabled></label>
    </div>
    <div class="section-label">成员</div>
    <div class="member-list">${allEntities.map((entity) => `
      <label class="member-option">
        <input type="checkbox" data-token-member="${escapeHtml(entity.id)}"
          ${token.member_ids.includes(entity.id) ? "checked" : ""}>
        <span>${escapeHtml(entity.name)} · ${escapeHtml(entity.id)}</span>
      </label>`).join("")}</div>`;
  const form = $("tokenInspector");
  form.querySelector("[data-token-name]").addEventListener("change", (event) => {
    token.name = event.target.value.trim() || token.id;
    markDirty();
    renderTokens();
    renderInspector();
  });
  form.querySelectorAll("[data-token-member]").forEach((input) => {
    input.addEventListener("change", () => {
      const id = input.dataset.tokenMember;
      if (input.checked && !token.member_ids.includes(id)) token.member_ids.push(id);
      if (!input.checked) token.member_ids = token.member_ids.filter((item) => item !== id);
      markDirty();
      renderTokens();
      renderInspector();
    });
  });
}

function deleteActive() {
  if (state.activeTokenId) {
    const id = state.activeTokenId;
    const token = tokenById(id);
    const candidateKey = token ? tokenCandidateKey(token) : null;
    state.annotation.style_tokens = state.annotation.style_tokens.filter((token) => token.id !== id);
    if (candidateKey) {
      const review = tokenReview();
      review.accepted_candidate_keys = review.accepted_candidate_keys.filter(
        (key) => key !== candidateKey,
      );
      review.ignored_candidate_keys.push(candidateKey);
    }
    state.activeTokenId = null;
    markDirty();
    renderAll();
    return;
  }
  const id = state.activeEntityId;
  const kind = entityKind(id);
  if (!kind) return;
  const parent = currentParent(id);
  if (kind === "element") {
    state.annotation.elements = state.annotation.elements.filter((item) => item.id !== id);
  } else {
    state.annotation.groups = state.annotation.groups.filter((item) => item.id !== id);
    state.annotation.layouts = state.annotation.layouts.filter((item) => item.target_id !== id);
    state.annotation.tree
      .filter((edge) => edge.parent_id === id)
      .forEach((edge) => { edge.parent_id = parent; });
  }
  state.annotation.tree = state.annotation.tree.filter((edge) => edge.child_id !== id);
  state.annotation.style_tokens.forEach((token) => {
    token.member_ids = token.member_ids.filter((memberId) => memberId !== id);
  });
  state.selectedEntityIds.delete(id);
  state.activeEntityId = null;
  normalizeAnnotation();
  markDirty();
  renderAll();
}

function sourceText(nodeIds) {
  if (!nodeIds?.length) return "无源节点";
  return nodeIds.map((id) => `#${id}`).join(", ");
}

function tokenSourceIds(token) {
  return [...new Set((token?.members || []).flatMap((member) => member.source_node_ids || []))];
}

function differenceItems() {
  const differences = state.adjudication?.record?.differences;
  if (!differences) return [];
  const items = [];
  differences.elements.only_human.forEach((item) => items.push({
    category: "elements", kind: "仅人工", title: item.human.name,
    detail: `${item.human.type} · ${item.human.text || "无文本"}`,
    nodeIds: item.source_node_ids, raw: item,
  }));
  differences.elements.only_ai.forEach((item) => items.push({
    category: "elements", kind: "仅 AI", title: item.ai.name,
    detail: `${item.ai.type} · ${item.ai.text || "无文本"}`,
    nodeIds: item.source_node_ids, raw: item,
  }));
  differences.elements.matched_but_changed.forEach((item) => items.push({
    category: "elements", kind: "元素属性", title: `${item.human.name} ↔ ${item.ai.name}`,
    detail: `${item.human.type} / ${item.ai.type} · IoU ${Number(item.bbox_iou).toFixed(2)}`,
    nodeIds: item.source_node_ids, raw: item,
  }));
  differences.groups.forEach((item) => items.push({
    category: "groups", kind: "语义分组",
    title: `${item.human.map((group) => group.name).join("、") || "人工无对应"} ↔ ${item.ai.map((group) => group.name).join("、") || "AI 无对应"}`,
    detail: `${item.human.map((group) => group.role).join("、") || "—"} / ${item.ai.map((group) => group.role).join("、") || "—"}`,
    nodeIds: item.source_node_ids, raw: item,
  }));
  differences.tree.forEach((item) => items.push({
    category: "tree", kind: "父子层级", title: `子实体 ${item.child.kind}`,
    detail: `人工父级 ${item.human_parents.length} · AI 父级 ${item.ai_parents.length}`,
    nodeIds: item.child.source_node_ids || [], raw: item,
  }));
  differences.layouts.forEach((item) => items.push({
    category: "layouts", kind: "布局约束", title: `布局 ${sourceText(item.source_node_ids)}`,
    detail: `${item.human.map((layout) => layout.mode).join("、") || "—"} / ${item.ai.map((layout) => layout.mode).join("、") || "—"}`,
    nodeIds: item.source_node_ids, raw: item,
  }));
  differences.tokens.only_human.forEach((item) => items.push({
    category: "tokens", kind: "仅人工 Token", title: item.kind,
    detail: `${item.members.length} 个成员`, nodeIds: tokenSourceIds(item),
  }));
  differences.tokens.only_ai.forEach((item) => items.push({
    category: "tokens", kind: "仅 AI Token", title: item.kind,
    detail: `${item.members.length} 个成员`, nodeIds: tokenSourceIds(item),
  }));
  return items;
}

function entitySourceIds(annotation, entity) {
  if (!annotation || !entity) return [];
  if (Array.isArray(entity.source_node_ids)) return entity.source_node_ids;
  const elements = new Map(annotation.elements.map((item) => [item.id, item]));
  return [...new Set((entity.source_element_ids || [])
    .flatMap((id) => elements.get(id)?.source_node_ids || []))].sort((a, b) => a - b);
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function sourceSignature(nodeIds) {
  return [...(nodeIds || [])].sort((a, b) => a - b).join(",");
}

function entitiesForSources(annotation, kind, nodeIds) {
  if (!annotation) return [];
  const entities = kind === "element" ? annotation.elements : annotation.groups;
  const signature = sourceSignature(nodeIds);
  return entities.filter((entity) => (
    sourceSignature(entitySourceIds(annotation, entity)) === signature
  ));
}

function referenceKeyForEntity(annotation, entityId) {
  if (entityId === "page_root") return { kind: "root", source_node_ids: [] };
  const entity = [...annotation.elements, ...annotation.groups]
    .find((item) => item.id === entityId);
  if (!entity) return null;
  return {
    kind: Array.isArray(entity.source_node_ids) ? "element" : "group",
    source_node_ids: entitySourceIds(annotation, entity),
  };
}

function goldIdForKey(key) {
  if (!key || key.kind === "root") return "page_root";
  return entitiesForSources(state.annotation, key.kind, key.source_node_ids)[0]?.id || null;
}

function removeGoldElement(id) {
  state.annotation.elements = state.annotation.elements.filter((item) => item.id !== id);
  state.annotation.tree = state.annotation.tree.filter((edge) => edge.child_id !== id);
  state.annotation.style_tokens.forEach((token) => {
    token.member_ids = token.member_ids.filter((memberId) => memberId !== id);
  });
}

function removeGoldGroup(id) {
  const parent = currentParent(id);
  state.annotation.groups = state.annotation.groups.filter((item) => item.id !== id);
  state.annotation.layouts = state.annotation.layouts.filter((item) => item.target_id !== id);
  state.annotation.tree
    .filter((edge) => edge.parent_id === id)
    .forEach((edge) => { edge.parent_id = parent; });
  state.annotation.tree = state.annotation.tree.filter((edge) => edge.child_id !== id);
  state.annotation.style_tokens.forEach((token) => {
    token.member_ids = token.member_ids.filter((memberId) => memberId !== id);
  });
}

function updateGoldElement(target, reference) {
  const preservedId = target.id;
  const preservedTokenRefs = target.style_token_refs || [];
  Object.assign(target, cloneJson(reference), {
    id: preservedId,
    style_token_refs: preservedTokenRefs,
  });
}

function applyElementChoice(item, source) {
  const reference = entitiesForSources(
    state.references[source], "element", item.nodeIds,
  )[0] || null;
  const exact = entitiesForSources(state.annotation, "element", item.nodeIds);
  if (!reference) {
    exact.forEach((element) => removeGoldElement(element.id));
    normalizeAnnotation();
    return true;
  }

  const desiredNodeIds = new Set(reference.source_node_ids || item.nodeIds);
  const overlapping = state.annotation.elements.filter((element) => (
    element.source_node_ids.some((id) => desiredNodeIds.has(id))
  ));
  const target = exact[0] || overlapping[0];
  overlapping.filter((element) => element !== target)
    .forEach((element) => removeGoldElement(element.id));
  if (target) {
    updateGoldElement(target, reference);
  } else {
    const element = cloneJson(reference);
    element.id = uniqueId("e", state.annotation.elements);
    element.style_token_refs = [];
    state.annotation.elements.push(element);
    state.annotation.tree.push({
      parent_id: "page_root", child_id: element.id, order: state.annotation.tree.length,
    });
  }
  normalizeAnnotation();
  return true;
}

function updateGoldGroup(target, reference) {
  const preservedId = target.id;
  const preservedMembers = target.source_element_ids || [];
  Object.assign(target, cloneJson(reference), {
    id: preservedId,
    source_element_ids: preservedMembers,
  });
}

function applyGroupChoice(item, source) {
  const referenceAnnotation = state.references[source];
  const desired = entitiesForSources(referenceAnnotation, "group", item.nodeIds);
  const existing = entitiesForSources(state.annotation, "group", item.nodeIds);
  const referenceToGold = new Map();

  const desiredIds = new Set(desired.map((group) => group.id));
  const missingChild = desired.some((reference) => (
    referenceAnnotation.tree
      .filter((edge) => edge.parent_id === reference.id)
      .some((edge) => {
        if (desiredIds.has(edge.child_id)) return false;
        const key = referenceKeyForEntity(referenceAnnotation, edge.child_id);
        return !goldIdForKey(key);
      })
  ));
  if (missingChild) {
    toast(`请先采用${source === "human" ? "人工" : "AI"}方案中的相关元素或子分组`, true);
    return false;
  }

  desired.forEach((reference, index) => {
    let target = existing[index];
    if (!target) {
      target = cloneJson(reference);
      target.id = uniqueId("g", state.annotation.groups);
      target.source_element_ids = [];
      state.annotation.groups.push(target);
      state.annotation.layouts.push(defaultLayout(target.id));
      state.annotation.tree.push({
        parent_id: "page_root", child_id: target.id, order: state.annotation.tree.length,
      });
    } else {
      updateGoldGroup(target, reference);
    }
    referenceToGold.set(reference.id, target.id);
  });
  existing.slice(desired.length).forEach((group) => removeGoldGroup(group.id));

  desired.forEach((reference) => {
    const targetId = referenceToGold.get(reference.id);
    const childEdges = referenceAnnotation.tree
      .filter((edge) => edge.parent_id === reference.id)
      .sort((left, right) => left.order - right.order);
    childEdges.forEach((edge) => {
      const key = referenceKeyForEntity(referenceAnnotation, edge.child_id);
      const childId = referenceToGold.get(edge.child_id) || goldIdForKey(key);
      if (childId && childId !== targetId) setParent(childId, targetId);
    });
  });
  normalizeAnnotation();
  return true;
}

function applyTreeChoice(item, source) {
  const childId = goldIdForKey(item.raw.child);
  if (!childId) {
    toast("金标准中找不到该子实体，请先处理元素或分组差异", true);
    return false;
  }
  const parents = item.raw[`${source}_parents`] || [];
  if (parents.length !== 1) {
    toast("参考结果没有唯一父级，不能一键采用", true);
    return false;
  }
  const parentId = goldIdForKey(parents[0]);
  if (!parentId) {
    toast("金标准中找不到参考父分组，请先采用对应分组", true);
    return false;
  }
  return setParent(childId, parentId);
}

function layoutComparable(layout) {
  if (!layout) return null;
  return {
    mode: layout.mode,
    gap: Number(layout.gap),
    padding: (layout.padding || []).map(Number),
    primary_align: layout.primary_align,
    cross_align: layout.cross_align,
    horizontal_resize: layout.horizontal_resize,
    vertical_resize: layout.vertical_resize,
  };
}

function applyLayoutChoice(item, source) {
  const referenceAnnotation = state.references[source];
  const desired = item.raw[source] || [];
  if (!desired.length) {
    toast(`${source === "human" ? "人工" : "AI"}结果中没有该分组布局，请先处理分组差异`, true);
    return false;
  }
  let applied = 0;
  desired.forEach((referenceLayout) => {
    const referenceGroup = referenceAnnotation.groups
      .find((group) => group.id === referenceLayout.target_id);
    const target = referenceGroup
      ? entitiesForSources(
        state.annotation, "group", entitySourceIds(referenceAnnotation, referenceGroup),
      )[0]
      : null;
    if (!target) return;
    const layout = state.annotation.layouts.find((entry) => entry.target_id === target.id);
    const next = { ...cloneJson(referenceLayout), target_id: target.id };
    if (layout) Object.assign(layout, next);
    else state.annotation.layouts.push(next);
    applied += 1;
  });
  if (!applied) {
    toast("金标准中找不到对应分组，请先处理分组差异", true);
    return false;
  }
  normalizeAnnotation();
  return true;
}

function elementChoiceMatches(item, source) {
  const reference = entitiesForSources(
    state.references[source], "element", item.nodeIds,
  )[0] || null;
  const current = entitiesForSources(state.annotation, "element", item.nodeIds);
  if (!reference) return current.length === 0;
  if (current.length !== 1) return false;
  const element = current[0];
  return JSON.stringify({
    source_node_ids: [...element.source_node_ids].sort((a, b) => a - b),
    type: element.type, bbox: element.bbox, name: element.name,
    text: element.text,
  }) === JSON.stringify({
    source_node_ids: [...item.nodeIds].sort((a, b) => a - b),
    type: reference.type, bbox: reference.bbox, name: reference.name,
    text: reference.text,
  });
}

function groupChoiceMatches(item, source) {
  const desired = entitiesForSources(state.references[source], "group", item.nodeIds);
  const current = entitiesForSources(state.annotation, "group", item.nodeIds);
  if (desired.length !== current.length) return false;
  const values = (groups) => groups.map((group) => ({
    role: group.role, name: group.name, bbox: group.bbox,
  }));
  return JSON.stringify(values(current)) === JSON.stringify(values(desired));
}

function treeChoiceMatches(item, source) {
  const childId = goldIdForKey(item.raw.child);
  if (!childId) return false;
  const currentParentId = currentParent(childId);
  const currentKey = referenceKeyForEntity(state.annotation, currentParentId);
  const desired = item.raw[`${source}_parents`] || [];
  return desired.length === 1
    && currentKey?.kind === desired[0].kind
    && sourceSignature(currentKey.source_node_ids) === sourceSignature(desired[0].source_node_ids);
}

function layoutChoiceMatches(item, source) {
  const referenceAnnotation = state.references[source];
  const desired = item.raw[source] || [];
  if (!desired.length) return false;
  return desired.every((referenceLayout) => {
    const referenceGroup = referenceAnnotation.groups
      .find((group) => group.id === referenceLayout.target_id);
    const target = referenceGroup
      ? entitiesForSources(
        state.annotation, "group", entitySourceIds(referenceAnnotation, referenceGroup),
      )[0]
      : null;
    const current = target
      ? state.annotation.layouts.find((layout) => layout.target_id === target.id)
      : null;
    return JSON.stringify(layoutComparable(current))
      === JSON.stringify(layoutComparable(referenceLayout));
  });
}

function choiceMatches(item, source) {
  if (item.category === "elements") return elementChoiceMatches(item, source);
  if (item.category === "groups") return groupChoiceMatches(item, source);
  if (item.category === "tree") return treeChoiceMatches(item, source);
  if (item.category === "layouts") return layoutChoiceMatches(item, source);
  return false;
}

function currentChoice(item) {
  const human = choiceMatches(item, "human");
  const ai = choiceMatches(item, "ai");
  if (human && ai) return "人工 / AI 一致";
  if (human) return "当前：人工";
  if (ai) return "当前：AI";
  return "当前：已调整";
}

function applyDifferenceChoice(item, source) {
  if (isAdjudicationLocked()) {
    toast("该样本已完成审核，不能继续修改", true);
    return;
  }
  if (["elements", "groups"].includes(item.category)) {
    const kind = item.category === "elements" ? "element" : "group";
    const desired = entitiesForSources(state.references[source], kind, item.nodeIds);
    const removesEntity = desired.length === 0;
    if (removesEntity && !window.confirm("该选择会从金标准中删除对应实体，确认继续？")) return;
  }
  let applied = false;
  if (item.category === "elements") applied = applyElementChoice(item, source);
  if (item.category === "groups") applied = applyGroupChoice(item, source);
  if (item.category === "tree") applied = applyTreeChoice(item, source);
  if (item.category === "layouts") applied = applyLayoutChoice(item, source);
  if (!applied) return;
  markDirty();
  state.validationErrors = localValidation();
  renderAll();
  toast(`金标准已采用${source === "human" ? "人工" : "AI"}方案，请检查右侧校验`);
}

function goldEntityForSources(nodeIds) {
  const signature = [...nodeIds].sort((a, b) => a - b).join(",");
  return [...state.annotation.elements, ...state.annotation.groups].find((entity) => (
    entitySourceIds(state.annotation, entity).join(",") === signature
  ));
}

function focusDifference(item, index) {
  state.activeDifferenceIndex = index;
  state.selectedNodeIds = new Set(item.nodeIds || []);
  const entity = goldEntityForSources(item.nodeIds || []);
  state.activeEntityId = entity?.id || null;
  state.activeTokenId = null;
  renderDifferences();
  renderNodes();
  renderEntities();
  renderCanvas();
  renderInspector();
  const boxes = (item.nodeIds || []).map(nodeById).filter(Boolean).map((node) => node.bbox);
  if (boxes.length) {
    const box = bboxUnion(boxes);
    $("canvasViewport").scrollTo({
      left: Math.max(0, box.x * state.scale - 40),
      top: Math.max(0, box.y * state.scale - 40),
      behavior: "smooth",
    });
  }
}

function renderDifferences() {
  if (!isAdjudication() || !state.adjudication) return;
  const metrics = state.adjudication.record.metrics || {};
  $("metricSummary").innerHTML = [
    ["元素 F1", metrics.leaf_f1],
    ["分组 F1", metrics.group_bcubed_f1],
    ["父子 F1", metrics.parent_f1],
    ["布局 κ", metrics.layout_cohen_kappa],
  ].map(([label, value]) => `<div class="metric-item"><span>${label}</span><strong>${Number(value || 0).toFixed(3)}</strong></div>`).join("");
  const allItems = differenceItems();
  const indexedItems = allItems.map((item, index) => ({ item, index }));
  const visible = state.differenceFilter === "all"
    ? indexedItems
    : indexedItems.filter(({ item }) => item.category === state.differenceFilter);
  $("differenceCount").textContent = String(visible.length);
  $("differenceList").innerHTML = visible.map(({ item, index }) => `
    <div class="difference-row ${state.activeDifferenceIndex === index ? "active" : ""}">
      <button type="button" class="difference-focus" data-difference-index="${index}">
      <span class="difference-row-head">
        <span class="difference-kind">${escapeHtml(item.kind)}</span>
        <span class="difference-source">${escapeHtml(sourceText(item.nodeIds))}</span>
      </span>
      <span class="difference-title">${escapeHtml(item.title)}</span>
      <span class="difference-detail">${escapeHtml(item.detail)}</span>
      </button>
      ${item.category === "tokens" ? "" : `
        <div class="difference-choice" role="group" aria-label="金标准采用来源">
          <span class="choice-status">${escapeHtml(currentChoice(item))}</span>
          <button type="button" class="choice-button human ${choiceMatches(item, "human") ? "selected" : ""}"
            data-choice-index="${index}" data-choice-source="human">采用人工</button>
          <button type="button" class="choice-button ai ${choiceMatches(item, "ai") ? "selected" : ""}"
            data-choice-index="${index}" data-choice-source="ai">采用 AI</button>
        </div>`}
    </div>`).join("") || '<div class="empty-state compact">该类型没有差异</div>';
  $("differenceList").querySelectorAll("[data-difference-index]").forEach((row) => {
    row.addEventListener("click", () => {
      const index = Number(row.dataset.differenceIndex);
      focusDifference(allItems[index], index);
    });
  });
  $("differenceList").querySelectorAll("[data-choice-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.choiceIndex);
      applyDifferenceChoice(allItems[index], button.dataset.choiceSource);
    });
  });
}

function renderAdjudicationPanel() {
  const panel = $("adjudicationPanel");
  panel.classList.toggle("hidden", !isAdjudication());
  if (!isAdjudication() || !state.adjudication) return;
  const record = state.adjudication.record;
  const review = record.review || {};
  const locked = isAdjudicationLocked();
  $("reviewStatus").textContent = locked ? "已审核" : "待审核";
  $("reviewStatus").classList.toggle("complete", locked);
  $("reviewerName").value = review.reviewed_by || $("reviewerName").value || "";
  $("reviewNotes").value = review.notes || $("reviewNotes").value || "";
  $("reviewerName").disabled = locked;
  $("reviewNotes").disabled = locked;
  $("reviewMeta").textContent = locked
    ? `${review.reviewed_by} · ${review.reviewed_at}`
    : `${differenceItems().length} 项差异待复核`;
}

function renderModeChrome() {
  const adjudication = isAdjudication();
  const hasAdjudication = Boolean(state.assignment?.adjudication);
  const adjudicationOption = $("modeSelect").querySelector('option[value="adjudication"]');
  adjudicationOption.hidden = !hasAdjudication;
  adjudicationOption.disabled = !hasAdjudication;
  $("annotatorField").classList.toggle("hidden", adjudication);
  $("differencesTabButton").classList.toggle("hidden", !adjudication);
  $("referenceControls").classList.toggle("hidden", !adjudication);
  $("adjudicationPanel").classList.toggle("hidden", !adjudication);
  $("saveButton").textContent = adjudication ? "保存仲裁草稿" : "保存草稿";
  $("submitButton").textContent = adjudication ? "完成人工审核" : "提交完成";
  document.querySelector(".panel-tabs").classList.toggle("adjudication", adjudication);
  document.querySelector(".workspace").classList.toggle(
    "editor-locked", isEditorLocked(),
  );
  $("saveButton").disabled = isEditorLocked();
  $("submitButton").disabled = isEditorLocked();
}

function localValidation() {
  const errors = [];
  const sourceCounts = new Map();
  state.annotation.elements.forEach((element) => {
    if (!element.source_node_ids.length) errors.push(`${element.id} 没有源节点`);
    element.source_node_ids.forEach((id) => sourceCounts.set(id, (sourceCounts.get(id) || 0) + 1));
    if (element.bbox.width <= 0 || element.bbox.height <= 0) errors.push(`${element.id} 的 bbox 面积无效`);
  });
  [...sourceCounts.entries()].filter(([, count]) => count > 1)
    .forEach(([id]) => errors.push(`源节点 #${id} 被多个元素引用`));
  state.annotation.groups.forEach((group) => {
    const children = directChildIds(group.id);
    if (!children.length) errors.push(`${group.id} 至少需要一个直接设计子实体`);
    if (!group.source_element_ids.length) {
      errors.push(`${group.id} 必须覆盖至少一个后代原子元素`);
    }
  });
  state.annotation.style_tokens.forEach((token) => {
    if (new Set(token.member_ids).size < 2) errors.push(`${token.id} 至少需要两个成员`);
  });
  if (state.assignment?.token_annotation_mode === "candidate_review" && !isAnnotationLocked()) {
    const review = tokenReview();
    const reviewedKeys = [...(review.reviewed_candidate_keys || [])].sort();
    const currentKeys = tokenCandidates().map((candidate) => candidate.key).sort();
    if (review.status !== "reviewed" || JSON.stringify(reviewedKeys) !== JSON.stringify(currentKeys)) {
      errors.push("请在 Token 页签检查当前样式候选");
    }
  }
  if (!state.annotation.elements.length) errors.push("完整标注至少需要一个原子元素");
  return errors;
}

async function saveAnnotation(submit = false, silent = false) {
  if (!state.annotation) return false;
  if (isEditorLocked()) {
    if (!silent) toast("该样本为只读状态", true);
    return false;
  }
  normalizeAnnotation();
  state.validationErrors = localValidation();
  renderValidation();
  if (submit && state.validationErrors.length) {
    toast("请先修复右侧校验错误", true);
    return false;
  }
  $("saveButton").disabled = true;
  $("submitButton").disabled = true;
  try {
    if (isAdjudication()) {
      const url = apiUrl(`/api/adjudication/${state.sampleId}`)
        + (submit ? "?submit=1" : "");
      const { data } = await requestJson(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          draft: state.annotation,
          review: {
            reviewed_by: $("reviewerName").value.trim(),
            notes: $("reviewNotes").value.trim(),
          },
        }),
      });
      state.annotation = data.draft;
      state.adjudication.record = data.record;
      state.validationErrors = data.errors || [];
      state.dirty = false;
      state.assignment.adjudication.progress = data.progress;
      state.assignment.adjudication.sample_status[state.sampleId] = data.record.status;
      renderAll();
      if (!silent) {
        toast(data.reviewed ? "该页已完成人工审核" : "仲裁草稿已保存", submit && !data.reviewed);
      }
      return !submit || data.reviewed;
    }
    const url = apiUrl(`/api/annotation/${state.annotator}/${state.sampleId}`)
      + (submit ? "?submit=1" : "");
    const { data } = await requestJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.annotation),
    });
    state.annotation = data.annotation;
    state.validationErrors = data.errors || [];
    state.dirty = false;
    const status = state.annotation.provenance.status;
    state.assignment.sample_status[state.annotator][state.sampleId] = status;
    const statuses = Object.values(state.assignment.sample_status[state.annotator]);
    state.assignment.progress[state.annotator] = {
      complete: statuses.filter((value) => value === "complete").length,
      draft: statuses.filter((value) => value !== "complete").length,
      total: statuses.length,
    };
    renderAll();
    if (!silent) {
      toast(data.submitted ? "标注已提交完成" : "草稿已保存", submit && !data.submitted);
    }
    return !submit || data.submitted;
  } catch (error) {
    toast(error.message, true);
    return false;
  } finally {
    $("saveButton").disabled = isEditorLocked();
    $("submitButton").disabled = isEditorLocked();
  }
}

async function loadSample(sampleId) {
  if (state.dirty) await saveAnnotation(false, true);
  state.sampleId = sampleId;
  state.selectedNodeIds.clear();
  state.selectedEntityIds.clear();
  state.activeEntityId = null;
  state.activeTokenId = null;
  state.activeDifferenceIndex = null;
  state.validationErrors = [];
  state.adjudication = null;
  state.references = { human: null, ai: null };
  $("reviewerName").value = "";
  $("reviewNotes").value = "";
  $("showAllNodes").checked = !isAdjudication();
  $("sampleSelect").value = sampleId;
  try {
    const [graphResult, annotationResult] = await Promise.all([
      requestJson(apiUrl(`/api/sample/${sampleId}/graph`)),
      requestJson(apiUrl(isAdjudication()
        ? `/api/adjudication/${sampleId}`
        : `/api/annotation/${state.annotator}/${sampleId}`)),
    ]);
    state.graph = graphResult.data;
    if (isAdjudication()) {
      state.adjudication = annotationResult.data;
      state.annotation = annotationResult.data.draft;
      state.references = {
        human: annotationResult.data.human,
        ai: annotationResult.data.ai,
      };
    } else {
      state.annotation = annotationResult.data;
    }
    state.dirty = false;
    $("pageScreenshot").src = apiUrl(`/api/sample/${sampleId}/screenshot`);
    $("pageScreenshot").alt = `样本 ${sampleId} 的网页截图`;
    renderAll();
    window.requestAnimationFrame(fitCanvas);
  } catch (error) {
    toast(error.message, true);
  }
}

async function changeAnnotator(annotator) {
  if (state.dirty) await saveAnnotation(false, true);
  state.annotator = annotator;
  renderProgress();
  await loadSample(state.sampleId);
}

async function changeMode(mode) {
  if (mode === "adjudication" && !state.assignment.adjudication) {
    toast("当前标注包尚未启用差异仲裁", true);
    return;
  }
  if (state.dirty) await saveAnnotation(false, true);
  state.mode = mode;
  $("modeSelect").value = mode;
  activateTab(mode === "adjudication" ? "differences" : "nodes");
  renderModeChrome();
  renderProgress();
  await loadSample(state.sampleId);
}

function fitCanvas() {
  if (!state.graph) return;
  const viewport = $("canvasViewport");
  const availableWidth = Math.max(120, viewport.clientWidth - 48);
  const availableHeight = Math.max(120, viewport.clientHeight - 48);
  state.scale = clamp(Math.min(
    availableWidth / state.graph.canvas.width,
    availableHeight / state.graph.canvas.height,
  ), 0.1, 1);
  renderCanvas();
}

function changeZoom(delta) {
  state.scale = clamp(Math.round((state.scale + delta) * 10) / 10, 0.1, 2);
  renderCanvas();
}

function renderAll() {
  if (!state.annotation || !state.graph) return;
  renderSaveState();
  renderProgress();
  renderNodes();
  renderEntities();
  renderTokens();
  renderCanvas();
  renderInspector();
  renderValidation();
  renderDifferences();
  renderAdjudicationPanel();
  renderModeChrome();
}

function activateTab(name) {
  document.querySelectorAll(".panel-tab").forEach((item) => {
    item.classList.toggle("active", item.dataset.tab === name);
  });
  document.querySelectorAll(".tab-content").forEach((item) => {
    item.classList.toggle("active", item.id === `${name}Tab`);
  });
}

function bindStaticEvents() {
  document.querySelectorAll(".panel-tab").forEach((button) => {
    button.addEventListener("click", () => {
      activateTab(button.dataset.tab);
    });
  });
  $("nodeSearch").addEventListener("input", (event) => {
    state.nodeSearch = event.target.value;
    renderNodes();
  });
  $("createElement").addEventListener("click", createElement);
  $("createGroup").addEventListener("click", createGroup);
  $("createToken").addEventListener("click", createToken);
  $("completeTokenReview").addEventListener("click", completeTokenReview);
  $("deleteEntity").addEventListener("click", deleteActive);
  $("showAllNodes").addEventListener("change", renderCanvas);
  $("showLabels").addEventListener("change", renderCanvas);
  $("showHumanReference").addEventListener("change", renderCanvas);
  $("showAiReference").addEventListener("change", renderCanvas);
  $("differenceFilter").addEventListener("change", (event) => {
    state.differenceFilter = event.target.value;
    state.activeDifferenceIndex = null;
    renderDifferences();
  });
  $("zoomOut").addEventListener("click", () => changeZoom(-0.1));
  $("zoomIn").addEventListener("click", () => changeZoom(0.1));
  $("fitCanvas").addEventListener("click", fitCanvas);
  $("saveButton").addEventListener("click", () => saveAnnotation(false));
  $("submitButton").addEventListener("click", () => saveAnnotation(true));
  $("modeSelect").addEventListener("change", (event) => changeMode(event.target.value));
  $("annotatorSelect").addEventListener("change", (event) => changeAnnotator(event.target.value));
  $("sampleSelect").addEventListener("change", (event) => loadSample(event.target.value));
  $("previousSample").addEventListener("click", () => {
    const index = state.assignment.samples.findIndex((sample) => sample.sample_id === state.sampleId);
    const next = state.assignment.samples[(index - 1 + state.assignment.samples.length) % state.assignment.samples.length];
    loadSample(next.sample_id);
  });
  $("nextSample").addEventListener("click", () => {
    const index = state.assignment.samples.findIndex((sample) => sample.sample_id === state.sampleId);
    const next = state.assignment.samples[(index + 1) % state.assignment.samples.length];
    loadSample(next.sample_id);
  });
  $("pageScreenshot").addEventListener("load", fitCanvas);
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  window.addEventListener("resize", () => {
    window.clearTimeout(window.intentResizeTimer);
    window.intentResizeTimer = window.setTimeout(fitCanvas, 120);
  });
}

async function initialize() {
  bindStaticEvents();
  try {
    const { data } = await requestJson("/api/assignment");
    state.assignment = data;
    state.mode = data.adjudication ? "adjudication" : "annotation";
    state.annotator = data.annotators[0];
    $("modeSelect").value = state.mode;
    $("annotatorSelect").innerHTML = data.annotators
      .map((annotator) => `<option value="${escapeHtml(annotator)}">${escapeHtml(annotator)}</option>`)
      .join("");
    $("sampleSelect").innerHTML = data.samples
      .map((sample) => `<option value="${escapeHtml(sample.sample_id)}">${escapeHtml(sample.sample_id)}</option>`)
      .join("");
    state.sampleId = data.samples[0]?.sample_id;
    activateTab(state.mode === "adjudication" ? "differences" : "nodes");
    renderModeChrome();
    renderProgress();
    if (state.sampleId) await loadSample(state.sampleId);
  } catch (error) {
    toast(`初始化失败：${error.message}`, true);
  }
}

initialize();
