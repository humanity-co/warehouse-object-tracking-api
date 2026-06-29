FROM node:20-alpine

WORKDIR /app

COPY apps/web/package.json /app/package.json
COPY apps/web/tsconfig.json /app/tsconfig.json
COPY apps/web/next.config.mjs /app/next.config.mjs
COPY apps/web/next-env.d.ts /app/next-env.d.ts
COPY apps/web/app /app/app
COPY apps/web/components /app/components
COPY apps/web/lib /app/lib

RUN npm install

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3000"]

