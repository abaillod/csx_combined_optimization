# WP set 01
one coil per half field period, constant R curves. Order 2.
```python
c1.set('zc(0)', 0.4)
c1.set('zc(1)', 0.2)
c1.set('phic(0)',np.pi/4)
c1.set('phis(1)',np.pi/8)
cur1 = ScaledCurrent(Current(1), 1e4)
```

# WP set 02
Two coils per half field period, constant R curves. Order 2.
```python
Rvessel = 0.0254 * v.params['DR']/2 - 0.1

c1 = wp1( 128, 2, Rvessel )
c1.set('zc(0)', 0.4)
c1.set('zc(1)', 0.2)
c1.set('phic(0)',np.pi/4)
c1.set('phis(1)',np.pi/8)
cur1 = ScaledCurrent(Current(1), 1e4)

c2 = wp1( 128, 2, Rvessel )
c2.set('zc(0)', -0.4)
c2.set('zc(1)', 0.2)
c2.set('phic(0)',np.pi/4)
c2.set('phis(1)',np.pi/8)
cur2 = ScaledCurrent(Current(1), 1e4)
```

# WP set 03
Two additional PF coils. DOFS are free to be ellipses.
```python
c = OrientedCurveXYZFourier( 128, 1 )
c.set('z0', 0.85)
c.set('roll',np.pi/2)
c.set('xc(1)', 0.5)
c.set('zs(1)', 0.5)
cur = ScaledCurrent(Current(1), 1e5)

c.fix_all()
c.unfix('z0')
c.unfix('xc(1)')
c.unfix('zs(1)')

base_curve = [c]
base_current = [cur]
```

# WP set 04
Large number of WPs on inner side of cylindrical vessel. WP geometry is fixed.
```python
Rvessel = 0.0254 * v.params['DR']/2 - 0.05
base_curve = []
base_current = []
ncurve = 4
for z0 in [-0.4,-0.2,0,0.2,0.4]:
    for ii in range(ncurve):
        c = wp1( 64, 1, Rvessel )
        
        c.set('zc(0)', z0)
        c.set('zc(1)', 0.1)
        c.set('phic(0)',(ii+.5)*np.pi/(2*ncurve))
        c.set('phis(1)',np.pi/(4*(ncurve+1)))
        c.fix_all()
        base_curve.append(c)
        base_current.append(ScaledCurrent(Current(1), 1e4))
        
for c in base_current:
    c.unfix_all()
```