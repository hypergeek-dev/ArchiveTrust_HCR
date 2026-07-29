"""Shared review document viewer used by Desktop v1 and v2.

Desktop v1 is the reference implementation for review overlays. This widget preserves that
production path in one place: resolve/render the selected PDF page, map provider geometry into
the rendered page's scene coordinates, draw red evidence rectangles and source labels, fit the
view, and clear stale state on every render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QVBoxLayout, QWidget

from archivetrust.clients.desktop.pdf_render import PageRenderError, render_pdf_page
from archivetrust.clients.desktop.ui import muted
from archivetrust.review.geometry_validation import GeometryValidationStatus, validate_evidence_geometry
from archivetrust.review.packet import ReviewPacket

SourceLabeler = Callable[[int], str]


class ReviewDocumentViewer(QWidget):
    def __init__(self, *, initial_caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom = 1.0
        self.caption = muted(initial_caption)
        self.scene = QGraphicsScene(self)
        self.canvas = QGraphicsView(self.scene)
        self.canvas.setMinimumWidth(380)
        self.canvas.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.canvas.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.caption)
        layout.addWidget(self.canvas, stretch=1)

    def render_packet(
        self,
        packet: ReviewPacket,
        *,
        archive_path: Path | None,
        source_label: SourceLabeler,
    ) -> None:
        self._zoom = 1.0
        self.scene.clear()
        self.validation_results = {}

        geometry = self._first_page_geometry(packet)
        page_image = None
        page_width_pt = page_height_pt = None
        render_scale = 1.0

        if geometry is not None and archive_path is not None and archive_path.exists():
            page_no, _metadata = geometry
            try:
                image, page_width_pt, page_height_pt = render_pdf_page(archive_path, page_no)
                render_scale = image.width() / page_width_pt if page_width_pt else 1.0
                page_image = image
            except PageRenderError as exc:
                self.caption.setText(f"Could not render the original page: {exc}")
        elif archive_path is None:
            self.caption.setText(
                "The original file could not be located in this Workspace's acquisition history."
            )

        if page_image is not None:
            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(page_image))
            self.scene.addItem(pixmap_item)
            page_rect = pixmap_item.boundingRect()
            self.caption.setText("Recognised text and its position are shown over the original page.")
        else:
            page_width_pt, page_height_pt = 612.0, 792.0
            page_rect = self.scene.addRect(
                QRectF(0, 0, page_width_pt, page_height_pt),
                QPen(QColor("#33404f")),
                QBrush(QColor("#e9ecf2")),
            ).rect()
            self.caption.setText(
                "The original page could not be rendered here; evidence boxes are shown on a page outline."
            )

        highlight = QPen(QColor("#d06767"))
        highlight.setWidth(2)
        drawn_union: QRectF | None = None
        invalid_count = 0
        for index, candidate in enumerate(packet.candidates):
            for evidence in candidate.evidence:
                validation = validate_evidence_geometry(
                    candidate=candidate,
                    evidence=evidence,
                    archive_path=archive_path,
                    current_page=geometry[0] if geometry is not None else None,
                )
                self.validation_results[evidence.evidence_id] = validation
                if not validation.may_draw_overlay or validation.transformed_rectangle is None:
                    if validation.status not in {
                        GeometryValidationStatus.MISSING,
                        GeometryValidationStatus.WHOLE_PAGE,
                    }:
                        invalid_count += 1
                    continue
                x, y, width, height = validation.transformed_rectangle
                rect = QRectF(x, y, width, height)
                if validation.status is GeometryValidationStatus.COARSE_VALID:
                    coarse_highlight = QPen(highlight)
                    coarse_highlight.setStyle(Qt.PenStyle.DashLine)
                    self.scene.addRect(rect, coarse_highlight)
                else:
                    self.scene.addRect(rect, highlight)
                label = self.scene.addText(source_label(index))
                label.setDefaultTextColor(QColor("#1a1f28"))
                label.setPos(rect.x(), rect.y() - 18)
                drawn_union = rect if drawn_union is None else drawn_union.united(rect)
        if invalid_count:
            self.caption.setText("The recorded source position could not be validated for this reading.")
        self.canvas.fitInView(drawn_union or page_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def show_empty_state(self, text: str = "No uncertainty under review.") -> None:
        self._zoom = 1.0
        self.scene.clear()
        item = self.scene.addText(text)
        item.setDefaultTextColor(QColor("#1a1f28"))

    def zoom_by_factor(self, factor: float) -> None:
        self._zoom *= factor
        self.canvas.scale(factor, factor)

    @staticmethod
    def _first_page_geometry(packet: ReviewPacket) -> tuple[int, dict] | None:
        for candidate in packet.candidates:
            for evidence in candidate.evidence:
                if evidence.page is not None and evidence.bounding_box is not None:
                    return evidence.page, evidence.coordinate_metadata
        return None
