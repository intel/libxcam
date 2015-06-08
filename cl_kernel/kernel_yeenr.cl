/*
 * function: kernel_yeenr
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 * yeenr_config: y edge enhancement and noise reduction configuration
 */

typedef struct
{
    float           yee_gain;
    float           yee_threshold;
    float           ynr_gain;
} CLYeenrConfig;

__kernel void kernel_yeenr (__read_only image2d_t input, __write_only image2d_t output, uint vertical_offset, CLYeenrConfig yeenr_config)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    int X = get_global_size(0);
    int Y = get_global_size(1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 y_in, y_out, uv_in;
    float4 nn, mn, on, pn, qn;
    float4 nm, mm, om, pm, qm;
    float4 no, mo, oo, po, qo;
    float4 np, mp, op, pp, qp;
    float4 nq, mq, oq, pq, qq;

    // copy UV
    if(y % 2 == 0) {
        uv_in = read_imagef(input, sampler, (int2)(x, y / 2 + vertical_offset));
        write_imagef(output, (int2)(x, y / 2 + vertical_offset), uv_in);
    }

    if (x < 2 || y < 2 || x > (X - 3) || y > (Y - 3)) {
        y_in = read_imagef(input, sampler, (int2)(x, y));
        write_imagef(output, (int2)(x, y), y_in);
    }
    else {
        nn = read_imagef(input, sampler, (int2)(x - 2, y - 2));
        mn = read_imagef(input, sampler, (int2)(x - 1, y - 2));
        on = read_imagef(input, sampler, (int2)(x, y - 2));
        pn = read_imagef(input, sampler, (int2)(x + 1, y - 2));
        qn = read_imagef(input, sampler, (int2)(x + 2, y - 2));

        nm = read_imagef(input, sampler, (int2)(x - 2, y - 1));
        mm = read_imagef(input, sampler, (int2)(x - 1, y - 1));
        om = read_imagef(input, sampler, (int2)(x, y - 1));
        pm = read_imagef(input, sampler, (int2)(x + 1, y - 1));
        qm = read_imagef(input, sampler, (int2)(x + 2, y - 1));

        no = read_imagef(input, sampler, (int2)(x - 2, y));
        mo = read_imagef(input, sampler, (int2)(x - 1, y));
        oo = read_imagef(input, sampler, (int2)(x, y));
        po = read_imagef(input, sampler, (int2)(x + 1, y));
        qo = read_imagef(input, sampler, (int2)(x + 2, y));

        np = read_imagef(input, sampler, (int2)(x - 2, y + 1));
        mp = read_imagef(input, sampler, (int2)(x - 1, y + 1));
        op = read_imagef(input, sampler, (int2)(x, y + 1));
        pp = read_imagef(input, sampler, (int2)(x + 1, y + 1));
        qp = read_imagef(input, sampler, (int2)(x + 2, y + 1));

        nq = read_imagef(input, sampler, (int2)(x - 2, y + 2));
        mq = read_imagef(input, sampler, (int2)(x - 1, y + 2));
        oq = read_imagef(input, sampler, (int2)(x, y + 2));
        pq = read_imagef(input, sampler, (int2)(x + 1, y + 2));
        qq = read_imagef(input, sampler, (int2)(x + 2, y + 2));

        float edgeV = (nm.x - om.x + qm.x - no.x - 14.0 * mo.x + 28.0 * oo.x - 14.0 * po.x - qo.x + np.x - op.x + qp.x) * 255.0 / 16.0;
        float edgeH = (mn.x - on.x + pn.x - 14.0 * om.x - mo.x + 28.0 * oo.x - po.x - 14.0 * op.x + mq.x - oq.x + pq.x) * 255.0 / 16.0;
        float edgeAll = (12.0 * oo.x - mm.x - 2.0 * om.x - pm.x - 2.0 * mo.x - 2.0 * po.x - mp.x - 2.0 * op.x - pp.x) * 255.0 / 8.0;
        float noiseAll = (32.0 * oo.x - nn.x - mn.x - on.x - pn.x - qn.x - nm.x - 2.0 * mm.x - 2.0 * om.x - 2.0 * pm.x - qm.x - no.x - 2.0 * mo.x - 2.0 * po.x - qo.x - np.x - 2.0 * mp.x - 2.0 * op.x - 2.0 * pp.x - qp.x - nq.x - mq.x - oq.x - pq.x - qq.x) * 255.0 / 32.0;
        float noiseV = edgeH;
        float noiseH = edgeV;

        float dirV = (fabs(2.0 * mm.x - mn.x - mo.x) + fabs(2.0 * om.x - on.x - oo.x) + fabs(2.0 * pm.x - pn.x - po.x) + fabs(2.0 * mo.x - mm.x - mp.x) + fabs(2.0 * oo.x - om.x - op.x) + fabs(2.0 * po.x - pm.x - pp.x) + fabs(2.0 * mp.x - mo.x - mq.x) + fabs(2.0 * op.x - oo.x - oq.x) + fabs(2.0 * pp.x - po.x - pq.x)) * 255.0;
        float dirH = (fabs(2.0 * mm.x - nm.x - om.x) + fabs(2.0 * om.x - mm.x - pm.x) + fabs(2.0 * pm.x - om.x - qm.x) + fabs(2.0 * mo.x - no.x - oo.x) + fabs(2.0 * oo.x - mo.x - po.x) + fabs(2.0 * po.x - oo.x - qo.x) + fabs(2.0 * mp.x - np.x - op.x) + fabs(2.0 * op.x - mp.x - pp.x) + fabs(2.0 * pp.x - op.x - qp.x)) * 255.0;
        float dirA = (fabs(2.0 * oo.x - om.x - op.x) + fabs(2.0 * oo.x - mo.x - po.x) + fabs(2.0 * oo.x - mm.x - pp.x) + fabs(2.0 * oo.x - pm.x - mp.x)) * 255.0;

        float edgeDir = dirH < (dirV < dirA ? dirV : dirA) ? edgeH  : (dirV < dirA ? edgeV : edgeAll);
        float noiseDir = dirH < (dirV < dirA ? dirV : dirA) ? noiseH  : (dirV < dirA ? noiseV : noiseAll);
        noiseDir = noiseDir * yeenr_config.ynr_gain;
        if(fabs(noiseAll) > 0.125)
            noiseDir = 0;
        edgeDir = fabs(edgeDir * yeenr_config.yee_gain) < yeenr_config.yee_threshold ?  edgeDir * yeenr_config.yee_gain : fabs(edgeDir) / edgeDir * yeenr_config.yee_threshold;

        y_out.x = (oo.x * 255.0 + edgeDir - noiseDir) / 255.0;
        y_out.y = 0.0;
        y_out.z = 0.0;
        y_out.w = 1.0;
        write_imagef(output, (int2)(x, y), y_out);
    }
}
