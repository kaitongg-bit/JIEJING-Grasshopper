"""Six authored, deterministic whole-window studies for the no-AI preset route.

Inspired by spatial devices in Suzhou garden lattice windows, not measured
copies of particular historic windows. White is open view; black is cut material.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

SIZE = 1200
NAMES = ("huiwen", "binglie", "haitang", "guibei", "fangsheng", "yulin")


def render_preset(name: str, inner: np.ndarray, spec: dict, bar: int, pitch: int,
                  seed: int = 7, open_center: bool = True) -> np.ndarray:
    if name not in NAMES:
        raise ValueError(f"No curated whole-window preset for {name}")
    x0, y0 = spec["origin_px"]
    ww, hh = spec["pixel_size"]
    cx, cy = x0 + ww / 2, y0 + hh / 2
    rx, ry = ww * .43, hh * .43
    def xy(x, y): return (round(cx + x * rx), round(cy + y * ry))
    def line(dst, a, b, width=bar): cv2.line(dst, xy(*a), xy(*b), 255, width, cv2.LINE_8)
    def path(dst, pts, width=bar, closed=False):
        cv2.polylines(dst, [np.array([xy(*p) for p in pts], np.int32)], closed, 255, width, cv2.LINE_8)
    def curve(dst, fn, width=bar, n=240, closed=True):
        path(dst, [fn(t) for t in np.linspace(0, 2 * math.pi, n, endpoint=not closed)], width, closed)
    def ellipse_mask(xc, yc, a, b):
        m=np.zeros_like(inner)
        curve(m,lambda t:(xc+a*math.cos(t),yc+b*math.sin(t)),1)
        cv2.floodFill(m,None,xy(xc,yc),255)
        return m
    stroke=np.zeros_like(inner)
    aperture=np.zeros_like(inner)
    # Each preset's viewing aperture and surrounding rhythm are authored together.
    if name == "huiwen":
        # Corner steps act as brackets; a broad quiet centre preserves the view.
        pts=[(-.55,-.48),(-.40,-.48),(-.40,-.58),(.40,-.58),(.40,-.48),(.55,-.48),(.55,.48),(.40,.48),(.40,.58),(-.40,.58),(-.40,.48),(-.55,.48)]
        path(aperture,pts,1,True);cv2.floodFill(aperture,None,xy(0,0),255)
        for sx in (-1,1):
            for sy in (-1,1):
                for k in range(3):
                    d=.085*k
                    p=[(sx*.99,sy*(.20+d)),(sx*(.68+d),sy*(.20+d)),
                       (sx*(.68+d),sy*(.68+d)),(sx*(.25+d),sy*(.68+d)),
                       (sx*(.25+d),sy*.99)]
                    path(stroke,p,max(9,bar-2))
    elif name == "binglie":
        # Ice crack is the irregular field itself.  A borrowed-view opening is
        # optional and must be requested explicitly; it is not intrinsic to
        # the ice-crack grammar.
        def borrowed_begonia(t, scale=1.0):
            r = 1 + .14 * math.cos(4 * t)
            return (.11 + .49 * scale * r * math.cos(t),
                    -.015 + .62 * scale * r * math.sin(t))
        if open_center:
            curve(aperture, borrowed_begonia, 1)
            cv2.floodFill(aperture, None, xy(.11, -.015), 255)
        rng=np.random.default_rng(seed)
        subdiv=cv2.Subdiv2D((0,0,SIZE,SIZE))
        crack_count = max(52, round(126 * (140 / pitch) ** 2))
        for xx,yy in rng.uniform(90,1110,(crack_count,2)):
            if not open_center or aperture[round(yy),round(xx)]==0:
                subdiv.insert((float(xx),float(yy)))
        facets,_=subdiv.getVoronoiFacetList([])
        for facet in facets:
            cv2.polylines(stroke,[np.rint(facet).astype(np.int32)],True,255,max(9,bar-5),cv2.LINE_8)
        # The cracks terminate directly on one clean rim.  Extra echo lines
        # made the opening look diagrammatic instead of architectural.
    elif name == "haitang":
        # A repeated four-lobed tracery field, after the photographed
        # Wangjuantang window. The central blossom is an enlarged viewing bay
        # in the same grammar, not a disconnected decorative badge.
        def blossom(t, xc, yc, a, b):
            r = 1 + .14 * math.cos(4 * t)
            return xc + a * r * math.cos(t), yc + b * r * math.sin(t)
        if open_center:
            curve(aperture, lambda t: blossom(t, 0, 0, .50, .57), 1)
            cv2.floodFill(aperture, None, xy(0, 0), 255)
        else:
            # Keep the borrowed view to one side. The personal symbol occupies
            # the opposite lattice field instead of being placed in this hole.
            curve(aperture, lambda t: blossom(t, -.43, 0, .37, .53), 1)
            cv2.floodFill(aperture, None, xy(-.43, 0), 255)
        cell_x, cell_y = .30 * pitch / 140, .40 * pitch / 140
        radius_x, radius_y = cell_x * .63, cell_y * .59
        for x in np.arange(-1.05, 1.06, cell_x):
            for y in np.arange(-1.12, 1.13, cell_y):
                curve(stroke, lambda t, x=x, y=y: blossom(t, x, y, radius_x, radius_y),
                      max(8, bar - 4), 100)
        # Slight overlaps at the tips join adjacent curves without a square grid.
    elif name == "guibei":
        aperture=ellipse_mask(0,0,.48,.69)
        # Offset hexagonal cells, sharing their edges.
        radius=.185 * pitch / 140
        for col,x in enumerate(np.arange(-1.10,1.11,radius*1.5)):
            for y in np.arange(-1.2+(col%2)*radius*math.sqrt(3)/2,1.2,radius*math.sqrt(3)):
                pts=[(x+radius*math.cos(k*math.pi/3),y+radius*math.sin(k*math.pi/3)) for k in range(6)]
                path(stroke,pts,max(9,bar-4),True)
        # Field is trimmed at the aperture; an uninterrupted rim is added below.
    elif name == "fangsheng":
        pts=[(0,-.70),(.62,0),(0,.70),(-.62,0)]
        path(aperture,pts,1,True);cv2.floodFill(aperture,None,xy(0,0),255)
        # Alternating diamonds make a woven side screen, without square backing.
        for y in np.arange(-1.02,1.03,.30):
            for col,x in enumerate(np.arange(-1.08,1.09,.31)):
                yy=y+(col%2)*.15
                path(stroke,[(x,yy-.17),(x+.155,yy),(x,yy+.17),(x-.155,yy)],max(9,bar-5),True)
        for sx in (-1,1): line(stroke,(sx*.80,0),(sx*.99,0),bar)
    elif name == "yulin":
        aperture=ellipse_mask(0,.02,.62,.45)
        # Scale arcs become shallow roof/eave bands with a broad low opening.
        for row,y in enumerate(np.arange(-1.08,1.12,.205)):
            offset=.19 if row%2 else 0
            for x in np.arange(-1.2+offset,1.3,.38):
                pts=[(x+.225*math.cos(t),y+.145*math.sin(t)) for t in np.linspace(math.pi,2*math.pi,34)]
                path(stroke,pts,max(9,bar-6))
        for sx in (-1,1):line(stroke,(sx*.80,.02),(sx*.99,.02),bar)
    # Opening boundary is load-bearing; keep a clean margin between it and field.
    # Give fields a definite architectural boundary. Pattern cells terminate
    # on a side or eave rail rather than at arbitrary distances from the view.
    if name in ("guibei", "fangsheng"):
        zone=np.zeros_like(inner)
        edge_x=round(.60*rx)
        zone[:,:round(cx-edge_x)]=255
        zone[:,round(cx+edge_x):]=255
        stroke=cv2.bitwise_and(stroke,zone)
        for sign in (-1,1):
            line(stroke,(sign*.60,-1.13),(sign*.60,1.13),bar)
            if name == "guibei":
                for yy in (-.46,0,.46):
                    rim_x=.48*math.sqrt(1-(yy/.69)**2)
                    line(stroke,(sign*(rim_x-.01),yy),(sign*.60,yy),bar)
            else:
                line(stroke,(sign*.60,0),(sign*.99,0),bar)
        if name == "guibei":
            for sign in (-1,1):
                line(stroke,(0,sign*.68),(0,sign*1.13),bar)
    elif name == "yulin":
        zone=np.zeros_like(inner)
        edge_y=round(.62*ry)
        zone[:round(cy-edge_y),:]=255
        zone[round(cy+edge_y):,:]=255
        stroke=cv2.bitwise_and(stroke,zone)
        for sign in (-1,1):
            line(stroke,(-.99,sign*.62),(.99,sign*.62),bar)
            for xx in (-.36,0,.36): line(stroke,(xx,sign*.43),(xx,sign*.62),bar)
    else:
        stroke=cv2.bitwise_and(stroke,cv2.bitwise_not(aperture))
    edge_size = max(8, round(bar * .72)) if name == "binglie" else bar + 1
    edge=cv2.subtract(cv2.dilate(aperture,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(edge_size,edge_size))),
                      cv2.erode(aperture,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(edge_size,edge_size))))
    result=cv2.bitwise_or(stroke,edge)
    return cv2.bitwise_and(result,inner)

